import asyncio
import logging
from typing import Dict, Any, Optional
from playwright.async_api import async_playwright, Page, Browser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WebVoyagerBrowserEnv:
    """
    Playwright-based browser environment wrapper for WebVoyager tasks.
    Extracts DOM accessibility trees, screenshots, and HTTP status codes at each step t.
    """
    def __init__(self, headless: bool = True, viewport_size: Dict[str, int] = None):
        self.headless = headless
        self.viewport = viewport_size or {"width": 1280, "height": 800}
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.last_http_status = 200
        self.error_logs = []

    async def initialize(self):
        """Initializes the Playwright browser instance."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        context = await self.browser.new_context(viewport=self.viewport)
        self.page = await context.new_page()

        self.page.on("response", lambda response: self._capture_response(response))
        self.page.on("pageerror", lambda err: self.error_logs.append(str(err)))

    def _capture_response(self, response):
        """Captures HTTP response statuses to detect 404s or server errors."""
        if response.url == self.page.url:
            self.last_http_status = response.status

    async def reset(self, start_url: str) -> Dict[str, Any]:
        """Resets the environment to a starting URL and returns initial observation o_1."""
        self.error_logs.clear()
        self.last_http_status = 200
        try:
            response = await self.page.goto(start_url, timeout=30000)
            if response:
                self.last_http_status = response.status
        except Exception as e:
            self.error_logs.append(f"Navigation timeout/error: {str(e)}")
            self.last_http_status = 500

        return await self.get_observation()

    async def get_observation(self) -> Dict[str, Any]:
        """
        Captures the current observation o_t comprising DOM text, screenshot bytes, and error states.
        """
        dom_text = ""
        try:
            dom_text = await self.page.evaluate("document.body.innerText")
        except Exception as e:
            self.error_logs.append(f"DOM extraction error: {str(e)}")

        screenshot_bytes = b""
        try:
            screenshot_bytes = await self.page.screenshot(full_page=False)
        except Exception as e:
            self.error_logs.append(f"Screenshot error: {str(e)}")

        observation = {
            "dom_text": dom_text.lower(),
            "screenshot": screenshot_bytes,
            "http_status_code": self.last_http_status,
            "error_logs": list(self.error_logs),
            "current_url": self.page.url if self.page else ""
        }
        
        self.error_logs.clear()
        return observation

    async def step(self, action_string: str) -> Dict[str, Any]:
        """
        Executes a discrete web action (e.g., CLICK [id], TYPE [id] [text], SCROLL) on the browser.
        """
        action_string = action_string.strip()
        logging.info(f"Executing Browser Action: {action_string}")

        try:
            if action_string.startswith("CLICK"):
                selector = action_string.split("[")[1].split("]")[0]
                await self.page.click(f"[data-qa-id='{selector}'], #{selector}, text={selector}", timeout=5000)
            elif action_string.startswith("TYPE"):
                parts = action_string.split("]")
                selector = parts[0].split("[")[1]
                text = parts[1].strip().strip('"').strip("'")
                await self.page.fill(f"[data-qa-id='{selector}'], #{selector}", text, timeout=5000)
            elif action_string.startswith("SCROLL"):
                await self.page.evaluate("window.scrollBy(0, 500)")
            elif action_string.startswith("FINISH"):
                logging.info("Task finished signal received.")
            else:
                self.error_logs.append(f"Unknown action format: {action_string}")
        except Exception as e:
            self.error_logs.append(f"Action execution failure [{action_string}]: {str(e)}")

        return await self.get_observation()

    async def close(self):
        """Closes the Playwright browser instance."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()