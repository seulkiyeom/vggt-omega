import sys, numpy as np, cv2, json
sys.path.insert(0, ".")
from a2_depth_quality import load_samples, project
S = load_samples(["long", "spatial"], 20, 3)
H = 256
rows = []
for k, s in enumerate(S):
    ua, va, za = project(s["K"][0], s["E"][0], s["ee"]); uw, vw, zw = project(s["K"][1], s["E"][1], s["ee"])
    d = s["depth"][0]; ui, vi = int(round(ua)), int(round(va))
    inb = 0 <= ui < H and 0 <= vi < H
    dz = float(za - d[vi, ui]) if inb else np.nan
    win = d[max(0, vi-6):vi+7, max(0, ui-6):ui+7] if inb else None
    rows.append(dict(k=k, suite=s["suite"], ua=ua, va=va, za=za, dz_center=dz, dz_min13=float(za - win.min()) if inb else np.nan, uw=uw, vw=vw, zw=zw))
    if k in (0, 7, 30):
        for v, (u, vv) in ((0, (ua, va)), (1, (uw, vw))):
            img = cv2.cvtColor(s["img"][v].copy(), cv2.COLOR_RGB2BGR)
            cv2.circle(img, (int(round(u)), int(round(vv))), 6, (0, 0, 255), 2)
            cv2.circle(img, (H-1-int(round(u)), H-1-int(round(vv))), 6, (0, 255, 0), 2)  # rot180 alt in green
            cv2.imwrite(f"/NHNHOME/nota/skyeom/omega_probes/a2/diag/ov_{k}_v{v}.png", cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST))
r = rows
print("agent proj u,v ranges:", np.percentile([x["ua"] for x in r], [5, 50, 95]).round(1), np.percentile([x["va"] for x in r], [5, 50, 95]).round(1))
print("agent z_ee median", np.nanmedian([x["za"] for x in r]).round(3), " signed dz (z_ee - depth@pixel): median", np.nanmedian([x["dz_center"] for x in r]).round(3), "p25/p75", np.nanpercentile([x["dz_center"] for x in r], [25, 75]).round(3))
print("agent z_ee - min depth in 13x13 window: median", np.nanmedian([x["dz_min13"] for x in r]).round(3), "p25/p75", np.nanpercentile([x["dz_min13"] for x in r], [25, 75]).round(3))
print("wrist proj u: median/std", np.median([x["uw"] for x in r]).round(1), np.std([x["uw"] for x in r]).round(2), " v: median/std", np.median([x["vw"] for x in r]).round(1), np.std([x["vw"] for x in r]).round(2), " z: median/std", np.median([x["zw"] for x in r]).round(3), np.std([x["zw"] for x in r]).round(3))
