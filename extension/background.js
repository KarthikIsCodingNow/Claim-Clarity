/**
 * ClaimClarity AI - Chromium Background Service Worker (Manifest V3)
 * Orchestrates viewport screen capture, API relay to local ML engine,
 * and cross-tab storage management.
 */

const API_BASE = "http://127.0.0.1:5001";

// Set initial badge state
chrome.runtime.onInstalled.addListener(() => {
  chrome.action.setBadgeBackgroundColor({ color: "#DB4A2B" });
  chrome.action.setBadgeText({ text: "" });
  console.log("[ClaimClarity AI] Extension installed successfully. Manifest V3 active.");
});

// Handle global keyboard command (Alt+C)
chrome.commands.onCommand.addListener(async (command) => {
  if (command === "activate_lens") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab && tab.id) {
      triggerLensOnTab(tab.id);
    }
  }
});

// Helper: trigger lens on a specific tab
async function triggerLensOnTab(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { action: "START_CIRCLE_TO_SEARCH" });
  } catch (err) {
    // Content script might not be injected yet (e.g. tab opened before extension loaded)
    console.warn("[ClaimClarity AI] Content script not responding, injecting dynamically...", err);
    try {
      await chrome.scripting.insertCSS({
        target: { tabId },
        files: ["content.css"]
      });
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["content.js"]
      });
      setTimeout(() => {
        chrome.tabs.sendMessage(tabId, { action: "START_CIRCLE_TO_SEARCH" });
      }, 100);
    } catch (injectErr) {
      console.error("[ClaimClarity AI] Failed to inject content script:", injectErr);
    }
  }
}

// Convert base64 dataURL to Blob
function dataURLtoBlob(dataurl) {
  const arr = dataurl.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  return new Blob([u8arr], { type: mime });
}

// Listen for messages from content scripts and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // 1. Activate Lens on active tab (triggered from popup)
  if (request.action === "ACTIVATE_LENS") {
    chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (tab && tab.id) {
        triggerLensOnTab(tab.id);
        sendResponse({ success: true });
      } else {
        sendResponse({ success: false, error: "No active tab" });
      }
    });
    return true; // Keep channel open for async response
  }

  // 2. Capture visible viewport screenshot
  if (request.action === "CAPTURE_VISIBLE_TAB") {
    chrome.tabs.captureVisibleTab(null, { format: "png" })
      .then((dataUrl) => {
        sendResponse({ success: true, dataUrl });
      })
      .catch((err) => {
        console.error("[ClaimClarity AI] Screenshot capture failed:", err);
        sendResponse({ success: false, error: err.message });
      });
    return true;
  }

  // 3. Relay image to local ClaimClarity ML Engine (bypasses webpage CSP)
  if (request.action === "ENHANCE_IMAGE") {
    (async () => {
      try {
        const imagePayload = request.imageBase64 || request.image_base64 || request.image || request.dataUrl;
        if (!imagePayload) {
          throw new Error("No image data provided for enhancement.");
        }

        // Default to classic_clahe as requested
        const targetMode = request.mode || "classic_clahe";

        const response = await fetch(`${API_BASE}/api/enhance`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            image_base64: imagePayload,
            imageBase64: imagePayload,
            image: imagePayload,
            mode: targetMode
          })
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.error || `Server responded with status ${response.status}`);
        }

        const result = await response.json();
        const meta = result.metadata || {};
        const dims = meta.dimensions ? [meta.dimensions.width, meta.dimensions.height] : (result.dimensions || [0, 0]);
        const damage = meta.damage_assessment || result.damage_assessment || {};

        // Cache the latest result for popup inspection
        const capturePayload = {
          timestamp: Date.now(),
          original: result.original_image,
          enhanced: result.enhanced_image,
          metadata: meta,
          dimensions: dims,
          damage_assessment: damage,
          processing_time_ms: meta.processing_time_ms || result.processing_time_ms || 120
        };

        await chrome.storage.local.set({ latest_claim_clarity_result: capturePayload });

        // Update extension badge to indicate ready
        chrome.action.setBadgeText({ text: "100%" });
        chrome.action.setBadgeBackgroundColor({ color: "#DB4A2B" });

        sendResponse({ success: true, result });
      } catch (err) {
        console.error("[ClaimClarity AI] ML Enhancement API relay failed:", err);
        sendResponse({ success: false, error: err.message });
      }
    })();
    return true;
  }

  // 4. Check backend health
  if (request.action === "CHECK_BACKEND_STATUS") {
    fetch(`${API_BASE}/api/health`, { method: "GET" })
      .then((res) => res.json())
      .then((data) => {
        sendResponse({ online: true, data });
      })
      .catch((err) => {
        sendResponse({ online: false, error: err.message });
      });
    return true;
  }

  // 5. Clear recent result
  if (request.action === "CLEAR_RECENT_RESULT") {
    chrome.storage.local.remove("latest_claim_clarity_result").then(() => {
      chrome.action.setBadgeText({ text: "" });
      sendResponse({ success: true });
    });
    return true;
  }
});
