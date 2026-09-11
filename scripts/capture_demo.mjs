import { createRequire } from "node:module";
import { mkdir } from "node:fs/promises";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.DEMO_URL || "http://127.0.0.1:8766";
await mkdir("docs/images", { recursive: true });
const browser = await chromium.launch({
  headless: true,
  ...(process.env.PLAYWRIGHT_CHANNEL
    ? { channel: process.env.PLAYWRIGHT_CHANNEL }
    : {}),
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  for (const [name, viewport] of Object.entries({
    desktop: { width: 1440, height: 1000 },
    mobile: { width: 390, height: 844 },
  })) {
    await page.setViewportSize(viewport);
    await page.goto(url);
    await page.getByRole("button", { name: "Generate drafts" }).click();
    await page.locator(".candidate").first().waitFor();
    assert.equal(await page.locator(".candidate").count(), 3);
    assert.match(
      await page.locator(".candidate .meta").first().innerText(),
      /^Rank score \d+\.\d{2} \/ Review required$/,
    );
    await page
      .getByRole("button", { name: "Approve candidate 1", exact: true })
      .click();
    await page.locator("#approval:not([hidden])").waitFor();
    assert.match(await page.locator("#status").innerText(), /No message sent/);
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      ),
      false,
    );
    await page.screenshot({
      path: `docs/images/demo-${name}.png`,
      fullPage: true,
    });
    for (const scenario of ["work", "unknown"]) {
      await page.selectOption("#scenario", scenario);
      await page.getByRole("button", { name: "Generate drafts" }).click();
      await page.locator(".candidate").first().waitFor();
      assert.equal(await page.locator(".candidate").count(), 3);
    }
    await page.getByRole("button", { name: "Reset session" }).click();
    await page.waitForFunction(
      () => document.querySelectorAll(".candidate").length === 0,
    );
    assert.equal(await page.locator("#approval").isVisible(), false);
  }
  assert.deepEqual(errors, []);
  console.log(
    "Desktop and mobile: all scenarios, local approval, reset, overflow and console checks passed.",
  );
} finally {
  await browser.close();
}
