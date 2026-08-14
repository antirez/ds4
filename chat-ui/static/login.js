(() => {
  const form = document.getElementById("loginForm");
  const username = document.getElementById("username");
  const password = document.getElementById("password");
  const errorEl = document.getElementById("loginError");
  const submitBtn = document.getElementById("loginBtn");

  function showError(message) {
    if (!errorEl) return;
    if (!message) {
      errorEl.hidden = true;
      errorEl.textContent = "";
      return;
    }
    errorEl.hidden = false;
    errorEl.textContent = message;
  }

  async function checkSession() {
    try {
      const res = await fetch("/api/auth/session");
      if (!res.ok) return;
      const data = await res.json();
      if (data && data.authenticated) {
        window.location.replace("/");
      }
    } catch {
      /* offline or server down */
    }
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    showError("");
    submitBtn.disabled = true;
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: username.value,
          password: password.value,
        }),
      });
      const text = await res.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        data = null;
      }
      if (!res.ok) {
        const msg =
          (data && data.error) ||
          (res.status === 401 ? "Invalid username or password." : "Sign in failed.");
        showError(msg);
        password.value = "";
        password.focus();
        return;
      }
      window.location.replace("/");
    } catch (err) {
      showError(`Could not reach the server (${err.message}).`);
    } finally {
      submitBtn.disabled = false;
    }
  });

  checkSession();
  username.focus();
})();
