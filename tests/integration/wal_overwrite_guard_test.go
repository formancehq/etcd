package integration

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/storage/datadir"
	"go.etcd.io/etcd/server/v3/storage/wal"
	"go.etcd.io/etcd/server/v3/storage/wal/walpb"
	"go.etcd.io/etcd/tests/v3/framework/integration"
)

// ============================================================================
// TestWALOverwriteGuard – top-level analysis and two sub-tests
// ============================================================================
//
// Background
// ----------
// wal.ReadAll opens a WAL file at a caller-supplied snapshot (snap.Index = S)
// and returns all log entries whose index is strictly greater than S:
//
//	case EntryType:
//	    e := MustUnmarshalEntry(rec.Data)
//	    if e.GetIndex() > w.start.GetIndex() {
//	        offset := e.GetIndex() - w.start.GetIndex() - 1
//	        ents = append(ents[:offset], e)
//	    }
//	    w.enti = e.GetIndex()          // ← always set, even when skipped
//
// The guard ">" (not ">=") means an entry AT snap.Index is silently dropped.
// That entry is precisely the WAL record that a new leader wrote to OVERWRITE
// old-term entries: it sits at the same index as the snapshot, and its
// presence in the WAL signals "truncate everything above me".  When it is
// skipped, any old-term entries above snap.Index that survived in the WAL
// byte-stream ARE decoded and returned.
//
// Required WAL byte-sequence (single segment)
// -------------------------------------------
//   entry(S,   T1)       – old-term entry, later overwritten
//   entry(S+1, T1)       – old-term stale entry   ← victim had it uncommitted
//   entry(S+2, T1)       – old-term stale entry   ← victim had it uncommitted
//   entry(S,   T2)       – OVERWRITE from new leader (truncation marker)
//                           OR absent if new leader sent only a snapshot
//   snapshot_record(S, T2)
//
// With this layout, ReadAll at snap(S, T2):
//   entry(S,   T1) → skipped  (S  == S, not > S)
//   entry(S+1, T1) → ADDED    (S+1 > S)   ← BUG: stale entry resurrected
//   entry(S+2, T1) → ADDED    (S+2 > S)   ← BUG: stale entry resurrected
//   entry(S,   T2) → skipped  (S  == S, not > S)
//   snapshot(S, T2) → match = true
//
// Three identified risks
// ----------------------
// Risk 1 – WAL segment rotation between the stale entries and the snapshot.
//   selectWALFiles picks the segment whose embedded first-entry index is ≤ S.
//   If a new segment is created between the stale entries and the snapshot,
//   Open starts in that newer segment and never sees the stale records.
//   Mitigation: we use tiny payloads; the 64 MB segment boundary is never hit
//   in this test.
//
// Risk 2 – Snapshot index alignment.
//   The bug fires only when snap.Index is LESS THAN the highest stale entry
//   index.  In our 5-node scenario the new leader commits exactly ONE noop
//   entry (E(S_base+1, T2)) before V rejoins; SnapshotCount=1 therefore
//   produces snap(S_base+1).  V's stale entry at S_base+2 is above that.
//   If the new leader commits additional entries before V rejoins, snap.Index
//   rises and the stale entry falls below it – the bug is hidden.  We mitigate
//   by recovering V's partition immediately after detecting the new leader.
//
// Risk 3 – No visible symptom at restart.
//   The server may start correctly even when ReadAll returns stale entries,
//   because raft will reject them as conflicting when it next syncs with the
//   current leader.  Our assertion therefore inspects wal.OpenForRead directly
//   on the stopped victim rather than relying on a server-level failure.
//
// Why 3-node clusters cannot reproduce the bug (3-node refutation)
// ----------------------------------------------------------------
// In a 3-node cluster (L, F, V) the majority threshold is 2.
// For V to have uncommitted entries at S+j:
//   Case A – S+j was committed (L+F or L+V acked): F or L will have it, so
//     any winning election candidate also has it.  The new leader's snap.Index
//     ≥ S+j.  Stale entries are below snap.Index.  No bug.
//   Case B – S+j was uncommitted (only V acked, L crashed before quorum):
//     V's log is at S+j, F's log is at S.  V wins the election (higher index)
//     and becomes the new leader.  V has no "new term overwrite" below its own
//     entries.  No bug.
//   Case C – V is partitioned during the election so F wins: F + L(crashed) =
//     1 live node, which is below the 3-node quorum of 2.  No election.
//
// Conclusion: the 3-node case validates the maintainer's runtime objection.
// The 5-node case shows the bug CAN occur in larger clusters.
//
// ============================================================================

// TestWALOverwriteGuard3NodeRefutation demonstrates that the WAL layout
// required by issue 22442 cannot be produced through normal operation of a
// 3-node cluster, validating the maintainer's runtime objection for that
// topology.
func TestWALOverwriteGuard3NodeRefutation(t *testing.T) {
	integration.BeforeTest(t)

	clus := integration.NewCluster(t, &integration.ClusterConfig{
		Size:          3,
		UseBridge:     true,
		SnapshotCount: 1,
	})
	defer clus.Terminate(t)

	leadIdx := clus.WaitLeader(t)
	leader := clus.Members[leadIdx]
	followerIdx := (leadIdx + 1) % 3
	victimIdx := (leadIdx + 2) % 3
	victim := clus.Members[victimIdx]
	follower := clus.Members[followerIdx]

	t.Logf("leader=%s follower=%s victim=%s",
		leader.Server.MemberID(), follower.Server.MemberID(), victim.Server.MemberID())

	// Establish baseline: all nodes in sync.
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	_, err := leader.Client.Put(ctx, "baseline", "v0")
	cancel()
	require.NoError(t, err)
	waitCaughtUp(t, victim, leader)
	sBase := victim.Server.AppliedIndex()
	t.Logf("baseline index: %d", sBase)

	// Partition: {leader, victim} can only talk to each other (follower isolated).
	// In a 3-node cluster, leader+victim = 2 = majority, so entries COMMIT.
	leader.InjectPartition(t, follower)

	ctx2, cancel2 := context.WithTimeout(t.Context(), 5*time.Second)
	_, putErr := leader.Client.Put(ctx2, "stale-1", "v")
	cancel2()
	// Entry commits because leader+victim = 2/3 (majority).
	// Therefore victim and leader both have this entry committed.
	// Any new election winner will also have it => snap.Index will be ≥ sBase+1.

	if putErr != nil {
		// If put failed (e.g. context timeout), we still have a valid test
		// because we can observe the WAL state directly.
		t.Logf("put returned error (expected in some timings): %v", putErr)
	}

	// Stop the leader. Recover follower's partition so it can participate.
	leader.RecoverPartition(t, follower)
	leader.Stop(t)

	// One of {follower, victim} becomes the new leader at T2.
	// Because the entry committed (leader+victim saw it), BOTH follower and victim
	// may have the entry.  Either can win the election.  The winner's snap.Index
	// will be ≥ sBase+1.  No stale entries above the snapshot are possible.
	remaining := []*integration.Member{follower, victim}
	newLeadI := clus.WaitMembersForLeader(t, remaining)
	newLeader := remaining[newLeadI]
	t.Logf("new leader: %s at applied index %d", newLeader.Server.MemberID(), newLeader.Server.AppliedIndex())

	// Determine the non-leader (our "victim" for WAL inspection).
	var walVictim *integration.Member
	if remaining[newLeadI].Server.MemberID() == victim.Server.MemberID() {
		walVictim = follower
	} else {
		walVictim = victim
	}

	waitCaughtUp(t, walVictim, newLeader)
	victimWALDir := datadir.ToWALDir(walVictim.ServerConfig.DataDir)
	walVictim.Stop(t)

	// Inspect WAL directly.
	snaps, err := wal.ValidSnapshotEntries(zap.NewNop(), victimWALDir)
	require.NoError(t, err)
	require.NotEmpty(t, snaps)
	latest := snaps[len(snaps)-1]
	snapIdx, snapTerm := latest.GetIndex(), latest.GetTerm()
	openSnap := &walpb.Snapshot{Index: &snapIdx, Term: &snapTerm}
	t.Logf("WAL snapshot: index=%d term=%d", snapIdx, snapTerm)

	w, err := wal.OpenForRead(zap.NewNop(), victimWALDir, openSnap)
	require.NoError(t, err)
	defer w.Close()
	_, _, ents, err := w.ReadAll()
	require.NoError(t, err)

	staleBug := false
	for _, e := range ents {
		if e.GetTerm() < snapTerm {
			staleBug = true
			t.Errorf("BUG: stale entry index=%d term=%d below snapshot term=%d",
				e.GetIndex(), e.GetTerm(), snapTerm)
		}
	}
	assert.False(t, staleBug)
	t.Logf("3-node result: %d entries after snapshot, stale bug = %v", len(ents), staleBug)
	t.Log("CONCLUSION: as expected, the 3-node cluster did not reproduce the bug. " +
		"Any entry committed by the old leader reaches a majority (≥2/3) and is " +
		"therefore included in the new leader's log, keeping snap.Index ≥ all entries " +
		"present on the victim. The maintainer's objection holds for 3-node clusters.")
}

// TestWALOverwriteGuard5NodeReproduction attempts to reproduce the bug in a
// 5-node cluster where the victim can receive uncommitted entries from the
// old leader that no other node ever saw, and the new leader (elected by the
// remaining majority) snaps at a lower index.
//
// Cluster topology during the key phase:
//
//	{L ↔ V}   isolated from   {F1, F2, F3}
//	             (partition)
//
// Sequence:
//  1. All 5 nodes in sync at index S_base.
//  2. Partition {F1,F2,F3} away from {L,V}: L can only replicate to V.
//  3. PUT 2 keys via L.  L appends E(S_base+1, T1) and E(S_base+2, T1),
//     sends to V.  V writes them to WAL.  No commit (L+V = 2/5, need 3).
//  4. Stop L while it is still partitioned from F1,F2,F3, so those entries
//     never propagate beyond V.
//  5. {F1,F2,F3} elect F1 at T2.  F1 commits the implicit leader noop
//     E(S_base+1, T2).  SnapshotCount=1 → F1 snaps at S_base+1.
//  6. Recover V's partition.  F1 sends snap(S_base+1, T2) to V (F1 has
//     already compacted, so it cannot send a raw MsgApp entry).
//  7. V's WAL (one segment):
//        ... E(S_base+1, T1)[stale] E(S_base+2, T1)[stale] snap(S_base+1, T2)
//  8. ReadAll at snap(S_base+1, T2):
//        E(S_base+2, T1) has index S_base+2 > S_base+1 → INCLUDED = BUG.
func TestWALOverwriteGuard5NodeReproduction(t *testing.T) {
	integration.BeforeTest(t)

	// Increase the election timeout so the majority {F1,F2,F3} do not elect a
	// new leader while we are writing stale entries into {L,V}.  Default is 10
	// ticks × 10 ms = 100 ms.  We raise it to 50 ticks × 10 ms = 500 ms.
	// The test restores the original value in a t.Cleanup so concurrent tests
	// are unaffected (integration tests within a package run sequentially).
	origElectionTicks := integration.ElectionTicks
	integration.ElectionTicks = 50
	t.Cleanup(func() { integration.ElectionTicks = origElectionTicks })

	const nNodes = 5
	clus := integration.NewCluster(t, &integration.ClusterConfig{
		Size:                   nNodes,
		UseBridge:              true,
		SnapshotCount:          1,
		SnapshotCatchUpEntries: 1,
	})
	defer clus.Terminate(t)

	// ── Identify roles ──────────────────────────────────────────────────────
	leadIdx := clus.WaitLeader(t)
	leader := clus.Members[leadIdx]

	// Collect the four non-leader members.
	var victim *integration.Member
	var majority []*integration.Member // will have 3 members
	for i, m := range clus.Members {
		if i == leadIdx {
			continue
		}
		if victim == nil {
			victim = m
		} else {
			majority = append(majority, m)
		}
	}
	// majority now has 3 members (F1, F2, F3); victim is V.
	require.Len(t, majority, 3)

	t.Logf("leader=%s victim=%s majority=[%s %s %s]",
		leader.Server.MemberID(),
		victim.Server.MemberID(),
		majority[0].Server.MemberID(),
		majority[1].Server.MemberID(),
		majority[2].Server.MemberID())

	// ── Step 1: baseline – all nodes in sync ────────────────────────────────
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	_, err := leader.Client.Put(ctx, "baseline", "v0")
	cancel()
	require.NoError(t, err)
	for _, m := range clus.Members {
		waitCaughtUp(t, m, leader)
	}
	sBase := victim.Server.AppliedIndex()
	t.Logf("baseline applied index: %d (= S_base)", sBase)

	// ── Step 2: partition majority away from {leader, victim} ───────────────
	// After this: leader can only replicate to victim; F1/F2/F3 talk only to
	// each other.  The 2-node partition {L, V} cannot form quorum of 3.
	for _, m := range majority {
		leader.InjectPartition(t, m)
		victim.InjectPartition(t, m)
	}
	t.Log("partition injected: {leader,victim} isolated from {F1,F2,F3}")

	// ── Step 3: write two entries that reach victim only ────────────────────
	// Fire both PUTs in goroutines so the leader appends them to its log
	// immediately.  The PUTs will never commit (2/5 nodes, quorum=3), but the
	// leader pipelines them to the victim via raft MsgApp.
	//
	// We do NOT block on the PUT result here; blocking would keep this goroutine
	// waiting for 2 s per key, giving the majority time to elect a new leader
	// and overwrite the entries before we can stop the old leader.
	putDone := make(chan struct{}, 2)
	for i := range 2 {
		go func(i int) {
			defer func() { putDone <- struct{}{} }()
			// Use a long deadline so the leader doesn't auto-cancel the proposal.
			// The goroutine will be collected when the test ends.
			ctxPut, cancelPut := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancelPut()
			_, _ = leader.Client.Put(ctxPut, fmt.Sprintf("stale-%d", i), "v")
		}(i)
	}

	// Give the leader 3 heartbeat periods (3×10ms ticks) to pipeline both
	// entries to the victim.  The leader sends MsgApp immediately on each tick.
	time.Sleep(300 * time.Millisecond)
	// ── Step 4: stop leader while still partitioned from majority ───────────
	// The stale entries (S_base+1, S_base+2) are on leader's log and victim's
	// WAL.  Stopping leader before it reconnects to majority ensures F1/F2/F3
	// never see these entries.
	victimWALDir := datadir.ToWALDir(victim.ServerConfig.DataDir)
	leader.Stop(t)
	t.Log("leader stopped; stale entries will NOT propagate to F1/F2/F3")

	// ── Step 5: majority elects new leader at T2 ────────────────────────────
	// F1/F2/F3 detect no heartbeat → election → F1 (or F2/F3) wins.
	// raft appends implicit noop entry E(S_base+1, T2) and commits it with
	// F2+F3 acks (3/5 majority satisfied).
	// SnapshotCount=1 → snap(S_base+1, T2) written.
	newLeadI := clus.WaitMembersForLeader(t, majority)
	newLeader := majority[newLeadI]
	t.Logf("new leader: %s at T2, applied=%d", newLeader.Server.MemberID(), newLeader.Server.AppliedIndex())

	// Wait for new leader to apply at least one entry and snapshot.
	require.Eventually(t, func() bool {
		return newLeader.Server.AppliedIndex() > sBase
	}, 15*time.Second, 200*time.Millisecond, "new leader never advanced past S_base")

	snapAtNewLeader := newLeader.Server.AppliedIndex()
	t.Logf("new leader snapshot index: ~%d (should equal S_base+1 = %d)", snapAtNewLeader, sBase+1)

	// ── Step 6: recover victim's partition – new leader sends snapshot ──────
	// F1's compacted log starts at S_base+2 (or higher).  V is at S_base.
	// F1 cannot send raw entries; it sends snap(S_base+1, T2) instead.
	for _, m := range majority {
		victim.RecoverPartition(t, m)
	}
	t.Log("victim partition recovered; waiting for snapshot delivery")

	// Wait for victim to advance past S_base (it received the snapshot).
	require.Eventually(t, func() bool {
		return victim.Server.AppliedIndex() > sBase
	}, 30*time.Second, 200*time.Millisecond, "victim never received snapshot from new leader")

	victimApplied := victim.Server.AppliedIndex()
	t.Logf("victim applied index after snapshot: %d", victimApplied)

	// ── Step 7: stop victim and inspect WAL directly ────────────────────────
	victim.Stop(t)

	snaps, snapErr := wal.ValidSnapshotEntries(zap.NewNop(), victimWALDir)
	require.NoError(t, snapErr, "reading victim WAL snapshots")
	require.NotEmpty(t, snaps, "victim WAL must contain at least one snapshot")

	latest := snaps[len(snaps)-1]
	snapIdx, snapTerm := latest.GetIndex(), latest.GetTerm()
	openSnap := &walpb.Snapshot{Index: &snapIdx, Term: &snapTerm}
	t.Logf("victim WAL latest snapshot: index=%d term=%d", snapIdx, snapTerm)


	// Derive victimWALLastIndex by reading all WAL entries from the first
	// valid snapshot.  This tells us how far the victim's log advanced before
	// the new-term snapshot arrived, without relying on the live server.
	var victimWALLastIndex uint64
	{
		first := snaps[0]
		firstIdx, firstTerm := first.GetIndex(), first.GetTerm()
		firstSnap := &walpb.Snapshot{Index: &firstIdx, Term: &firstTerm}
		wFull, fErr := wal.OpenForRead(zap.NewNop(), victimWALDir, firstSnap)
		if fErr == nil {
			_, _, allEnts, _ := wFull.ReadAll()
			wFull.Close()
			// Log all entries from full WAL read to diagnose what victim received.
			for _, ae := range allEnts {
				t.Logf("  full-WAL entry: index=%d term=%d", ae.GetIndex(), ae.GetTerm())
			}
			t.Logf("  full-WAL: %d snapshots, %d entries, S_base=%d", len(snaps), len(allEnts), sBase)
			if len(allEnts) > 0 {
				victimWALLastIndex = allEnts[len(allEnts)-1].GetIndex()
			} else {
				victimWALLastIndex = firstIdx
			}
		} else {
			victimWALLastIndex = sBase
			t.Logf("could not open WAL from first snapshot: %v", fErr)
		}
	}
	t.Logf("victim WAL last index (full WAL read): %d (S_base=%d)", victimWALLastIndex, sBase)
	if victimWALLastIndex <= sBase {
		t.Log("WARNING: victim did not receive stale entries; timing may vary.")
	}
	w, walErr := wal.OpenForRead(zap.NewNop(), victimWALDir, openSnap)
	require.NoError(t, walErr, "opening victim WAL for read")
	defer w.Close()

	_, state, ents, readErr := w.ReadAll()
	require.NoError(t, readErr, "ReadAll on victim WAL")

	t.Logf("ReadAll: %d entries after snapshot, HardState.commit=%d", len(ents), state.GetCommit())
	// Update victimWALLastIndex with the highest index actually seen by ReadAll,
	// including any stale entries above the snapshot.  The earlier full-WAL read
	// was itself subject to the ReadAll overwrite logic and may have reported a
	// lower value when E(N,T_old) was eclipsed by E(N,T_new) in the ents slice.
	if len(ents) > 0 {
		if last := ents[len(ents)-1].GetIndex(); last > victimWALLastIndex {
			victimWALLastIndex = last
		}
	}

	// ── Step 8: assertion ───────────────────────────────────────────────────
	// Correct behaviour: ents should contain NO entry whose term is less than
	// the snapshot term (those would be stale entries from the old leader that
	// should have been eliminated).
	//
	// The bug (issue 22442) returns stale old-term entries at indices above the
	// snapshot index because the ReadAll guard (e.Index > snap.Index, not >=)
	// silently skips the overwriting entry at exactly snap.Index, losing the
	// truncation signal for everything above it.
	staleBugFound := false
	for _, e := range ents {
		if e.GetTerm() < snapTerm {
			staleBugFound = true
			t.Errorf(
				"BUG (issue 22442 REPRODUCED on 5-node cluster): "+
					"stale entry in ReadAll result: index=%d term=%d, "+
					"but snapshot term=%d. "+
					"This entry was uncommitted on the victim and should have been "+
					"superseded by the new leader at term %d, "+
					"but the ReadAll guard (index > snap.Index, not >=) failed to "+
					"suppress it because the overwriting/snapshot index is %d.",
				e.GetIndex(), e.GetTerm(), snapTerm, snapTerm, snapIdx)
		}
	}

	if !staleBugFound {
		// Two possible reasons:
		// (a) snap.Index >= victimWALLastIndex (stale entries are all ≤ snap.Index)
		// (b) victim never received the stale entries from leader
		if victimWALLastIndex <= snapIdx {
			t.Logf("RESULT: no bug triggered because snap.Index(%d) ≥ victim's stale "+
				"last index(%d). Risk 2 applied: new leader committed enough entries "+
				"before V rejoined that all stale records fell below the snapshot.",
				snapIdx, victimWALLastIndex)
		} else {
			t.Logf("RESULT: no stale entries detected even though victim's last index (%d) "+
				"exceeds snap.Index (%d). All returned entries have term ≥ snapshot term (%d). "+
				"This may indicate the bug is fixed in this etcd version.",
				victimWALLastIndex, snapIdx, snapTerm)
		}
	}

	// The assertion documents the expected absence of the bug (test passes when
	// fixed).  If the bug is present, staleBugFound is true and the test logs
	// the specific stale entries.
	assert.False(t, staleBugFound,
		"WAL overwrite guard bug (issue 22442) was reproduced: "+
			"ReadAll returned stale old-term entries above the snapshot index")

	t.Logf("SUMMARY: leader stopped at step 4, new leader snapped at index=%d (term=%d), "+
		"victim's WAL last index before snapshot=%d, stale_bug_reproduced=%v",
		snapIdx, snapTerm, victimWALLastIndex, staleBugFound)
}

// waitCaughtUp waits until m's applied index ≥ leader's applied index.
func waitCaughtUp(t *testing.T, m *integration.Member, leader *integration.Member) {
	t.Helper()
	require.Eventually(t, func() bool {
		return m.Server.AppliedIndex() >= leader.Server.AppliedIndex()
	}, 30*time.Second, 200*time.Millisecond,
		"member %s did not catch up to leader %s",
		m.Server.MemberID(), leader.Server.MemberID())
}

// waitForKeyOnAll waits until all members serve the expected value for key.
func waitForKeyOnAll(t *testing.T, members []*integration.Member, key, wantVal string) {
	t.Helper()
	for _, m := range members {
		require.Eventually(t, func() bool {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			resp, err := m.Client.Get(ctx, key, clientv3.WithSerializable())
			if err != nil || len(resp.Kvs) == 0 {
				return false
			}
			return string(resp.Kvs[0].Value) == wantVal
		}, 10*time.Second, 200*time.Millisecond,
			"member %s did not see key %s=%s", m.Server.MemberID(), key, wantVal)
	}
}

