# Lessons

- When client threads publish server statistics, copy every reported value from worker-owned session state into mutex-protected server state; never read mutable `ds4_session` fields directly outside the single worker.
- When running supplemental AddressSanitizer checks on macOS, disable unsupported leak detection and time-box the run so an unexpectedly slow instrumented test cannot compete with the live Metal server.
- When a lifecycle claims an output adapter boundary, route prefill, stream-open, delta, flush, and terminal writes through that seam; terminal-only adaptation leaves irreversible failures outside the transaction.
- When reporting session disposition after a failed mutator, query explicit checkpoint validity; a nonzero token position does not prove the backend graph is reusable.
- When testing an ordered lifecycle, size the event log for the complete success path and activate asynchronous refresh behavior through a real producer, not by seeding its internal flag.
