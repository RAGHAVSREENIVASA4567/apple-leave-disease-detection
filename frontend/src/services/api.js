const API_URL = import.meta.env.VITE_API_URL || "http://localhost:4000";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}/api${path}`, options);
  } catch {
    throw new Error("Cannot reach the AppleCare AI API. Make sure the backend is running.");
  }

  if (!response.ok) {
    let message = "Something went wrong. Please try again.";
    try {
      const body = await response.json();
      if (body.error) message = body.error;
    } catch {
      // Keep the generic message if the server response is not JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

export function analyzeLeaf(file) {
  const formData = new FormData();
  formData.append("image", file);
  return request("/analyze-leaf", { method: "POST", body: formData });
}

export async function getHistory() {
  const data = await request("/history");
  return data.scans;
}

export function getApiHealth() {
  return request("/health");
}

export async function getHistoryItem(id) {
  const data = await request(`/history/${encodeURIComponent(id)}`);
  return data.scan;
}

export function deleteHistoryItem(id) {
  return request(`/history/${encodeURIComponent(id)}`, { method: "DELETE" });
}
