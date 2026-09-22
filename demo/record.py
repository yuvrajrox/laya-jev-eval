"""Record the demo as MP4 clips for social, driving the real page with Playwright.

Playwright only writes WebM, and X needs H.264 MP4, so each clip is transcoded with the
ffmpeg binary that ships inside imageio-ffmpeg (no system install).

A synthetic cursor is drawn into the page and animated toward each target before it is
clicked. A headless browser has no pointer, and a demo video where buttons fire with
nothing touching them reads as fake.
"""
import os, shutil, subprocess, sys, time
import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000"
W, H = 1280, 720
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")

CURSOR = """
(() => {
  if (document.getElementById('__cur')) return;
  const c = document.createElement('div'); c.id='__cur';
  c.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;z-index:99999;'+
    'pointer-events:none;transition:transform .45s cubic-bezier(.3,.7,.3,1);'+
    'background:no-repeat center/contain;filter:drop-shadow(0 2px 3px rgba(0,0,0,.6))';
  c.style.backgroundImage = "url(\\"data:image/svg+xml;utf8,"+
    encodeURIComponent('<svg xmlns=\\'http://www.w3.org/2000/svg\\' viewBox=\\'0 0 24 24\\'>'+
    '<path d=\\'M5 2l14 9-6 1.2 3.3 6.4-2.7 1.4L10.3 14 5 18z\\' fill=\\'white\\' stroke=\\'black\\' stroke-width=\\'1.2\\'/></svg>')+"\\")";
  document.body.appendChild(c);
  window.__moveCur = (x,y,click) => {
    c.style.transform = `translate(${x-3}px,${y-2}px) scale(${click?0.82:1})`;
  };
  window.__moveCur(640,360,false);
})();
"""


class Rec:
    def __init__(self, page):
        self.page = page

    def cursor_to(self, sel, settle=0.55):
        box = self.page.locator(sel).first.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.evaluate("([x,y])=>window.__moveCur(x,y,false)", [x, y])
        time.sleep(settle)
        return x, y

    def click(self, sel, settle=0.55, after=0.25):
        x, y = self.cursor_to(sel, settle)
        self.page.evaluate("([x,y])=>window.__moveCur(x,y,true)", [x, y])
        time.sleep(0.12)
        self.page.locator(sel).first.click()
        self.page.evaluate("([x,y])=>window.__moveCur(x,y,false)", [x, y])
        time.sleep(after)

    def select(self, sel, value, settle=0.5):
        self.cursor_to(sel, settle)
        self.page.select_option(sel, label=value) if False else self.page.select_option(sel, value=value)
        self.page.dispatch_event(sel, "change")
        time.sleep(0.3)


def transcode(webm, mp4):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-i", webm, "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-vf", "scale=%d:%d" % (W, H), mp4],
                   check=True, capture_output=True)
    return mp4


def clip_draw(pw, name):
    ctx = pw.chromium.launch(args=["--force-device-scale-factor=1"]).new_context(
        viewport={"width": W, "height": H}, device_scale_factor=2,
        record_video_dir=OUT, record_video_size={"width": W, "height": H})
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.evaluate(CURSOR)
    page.evaluate("window.scrollTo(0,0)")
    r = Rec(page)
    time.sleep(1.0)
    for preset in ["an orange fox with green eyes on a cream background",
                   "a pink panda with blue eyes on a lime background"]:
        r.select("#preset", preset)
        r.click("#drawBtn", after=0.2)
        page.wait_for_function("document.getElementById('res').textContent==='256'", timeout=60000)
        time.sleep(1.1)
    time.sleep(0.5)
    path = page.video.path()
    ctx.close()
    return path


def clip_classify(pw, name):
    ctx = pw.chromium.launch(args=["--force-device-scale-factor=1"]).new_context(
        viewport={"width": W, "height": H}, device_scale_factor=2,
        record_video_dir=OUT, record_video_size={"width": W, "height": H})
    page = ctx.new_page()
    page.goto(URL, wait_until="networkidle")
    page.evaluate(CURSOR)
    r = Rec(page)
    time.sleep(1.0)
    r.click("#clsBtn", after=0.2)
    page.wait_for_selector("#out.on", timeout=60000)
    time.sleep(2.4)
    r.select("#epreset", "Lovely to deal with a company that answers the phone. That alone.")
    r.click("#clsBtn", after=0.2)
    page.wait_for_selector("#out.on", timeout=60000)
    time.sleep(2.4)
    path = page.video.path()
    ctx.close()
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as pw:
        # warm the models so the first clip is not a loading screen
        import urllib.request, json as J
        for ep, body in (("draw", {"caption": "a purple cat with yellow eyes on a mint background"}),
                         ("classify", {"email": "hello"})):
            try:
                urllib.request.urlopen(urllib.request.Request(
                    URL + "/api/" + ep, data=J.dumps(body).encode(),
                    headers={"Content-Type": "application/json"}), timeout=300).read()
            except Exception as e:
                print("warmup %s: %s" % (ep, e))
        print("models warm")
        for fn, name in ((clip_draw, "01-draw"), (clip_classify, "02-decide")):
            webm = fn(pw, name)
            mp4 = os.path.join(OUT, name + ".mp4")
            transcode(webm, mp4)
            os.remove(webm)
            sz = os.path.getsize(mp4) / 1e6
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            dur = subprocess.run([ff, "-i", mp4], capture_output=True, text=True).stderr
            dur = [l for l in dur.split("\n") if "Duration" in l]
            print("  %-10s %.1f MB  %s" % (name, sz, dur[0].strip().split(",")[0] if dur else ""))
    print("\nclips -> %s" % OUT)
