# Lessons

- When client threads publish server statistics, copy every reported value from worker-owned session state into mutex-protected server state; never read mutable `ds4_session` fields directly outside the single worker.
