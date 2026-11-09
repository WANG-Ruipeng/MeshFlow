#!/usr/bin/env python3
"""Build an offline, complete gallery index after the frozen 56 PNGs exist.
Stdlib only. Does not render, score, modify RAW, or import the frozen evaluator.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
from urllib.parse import quote

LABELS = ("R2_FROZEN", "R2_CONT_OLD", "R2_CONT_USER")
EXPECTED_EVALUATION_SHA = "171c16b24d5079d85c9a60971a77e2c1b521ee9a67fca1f458d820e1f25cd23b"
METRICS = (
 ("CF_proper_affected_free_fraction","CF proper / free"),
 ("CF_illegal_affected_free_fraction","CF illegal / free"),
 ("FF_illegal_affected_free_fraction","FF illegal / free"),
 ("ConflictCov_C","I confirmed"),
 ("ConflictCov_C_upper","I upper"),
 ("free_minimum_angle_lt5_fraction","Free angle <5°"),
)
HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>R2 用户式条件微调：完整配对图集</title>
<style>
:root{color-scheme:light;--border:#d4dce5;--ink:#202b39;--muted:#536375}
*{box-sizing:border-box}body{margin:0;background:#eef2f6;color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1650px;margin:auto;padding:20px}h1{font-size:25px;margin:4px 0 12px}h2{font-size:17px;overflow-wrap:anywhere}
p{margin:8px 0}.note{border-left:4px solid #4178a8;background:#fff;padding:12px 16px}
.controls{position:sticky;top:0;z-index:2;background:#f8fafcf5;border:1px solid var(--border);padding:12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:18px 0}
label{display:flex;align-items:center;gap:5px}select,input,button{font:inherit;padding:6px 9px;border:1px solid #b5c2ce;border-radius:4px;background:white}
input{min-width:220px}button{cursor:pointer}a{color:#185b91}a:focus,button:focus,input:focus,select:focus{outline:3px solid #90c6f1}
.card{background:white;border:1px solid var(--border);border-radius:7px;padding:14px;margin:20px 0;break-inside:avoid}
.meta{color:var(--muted);overflow-wrap:anywhere}.badge{display:inline-block;padding:2px 8px;border-radius:4px;background:#e5edf5;margin-right:6px}
figure{margin:12px 0 0}figcaption{background:#f4f7fa;padding:6px 9px;font-weight:600}
img{width:100%;height:auto;display:block;background:white;border:1px solid #e3e8ee}
.links{font-size:13px;display:flex;gap:16px;flex-wrap:wrap}.method-order{font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:9px}th,td{padding:7px;border:1px solid var(--border);text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#f1f5f8}.score-wrap{overflow:auto}
details{margin:12px 0}summary{cursor:pointer}footer{color:var(--muted);font-size:13px;overflow-wrap:anywhere}
body[data-view="colored"] figure[data-mode="neutral"],body[data-view="neutral"] figure[data-mode="colored"]{display:none}
.card[hidden]{display:none}.empty{padding:30px;background:white;text-align:center}
@media(max-width:800px){main{padding:10px}.controls{position:static}h1{font-size:22px}input{min-width:150px}.card{padding:8px}}
@media print{.controls{position:static}details{display:block}.card{break-inside:avoid}body{background:white}}
</style></head>
<body data-view="colored"><main>
<h1>R2 用户式条件微调：完整配对图集</h1>
<p id="coverage"></p>
<div class="note">
<p>全部 <strong>28 个固定请求 × 3 个模型＝84 份 RAW</strong>，每请求有中性、蓝橙两张图，每张均包含三模型与两个视角。没有筛图、后处理或新增生成。</p>
<p><strong>蓝色＝原样保留的 C；橙色＝生成的 free。</strong> 模型列顺序保持 R2_FROZEN、R2_CONT_OLD、R2_CONT_USER。每个条件的全部模型与两份噪声共用完整显示边界；不同条件的显示尺度可不同，指标归一尺度保持预先冻结的 L。</p>
<p>seat/back、small/medium/full 是来源条件的组织标签，不是模型输入文本。full 仅指完整登记区域 R，合理侧壁、底面和厚度不自动算用途错误。OLDREG 没有新语义区域标签。</p>
<p><strong>AI_PROXY 为 AI 辅助技术审阅，不是人类评分或人类可用率。</strong> 应分别描述整体质量与 C 用途，并保留“改善／接近／退化／不确定”。S、E 和来源 GT 重建排名不适用于本轮。</p>
</div>
<div class="controls">
<label>面板 <select id="panel"><option value="">全部</option><option>NEWDEV</option><option>OLDREG</option></select></label>
<label>区域 <select id="region"><option value="">全部</option><option value="seat">seat</option><option value="back">back</option><option value="NA_OLD_GEOMETRIC">旧几何条件</option></select></label>
<label>显示 <select id="mode"><option value="colored">蓝 C／橙 free</option><option value="neutral">中性</option><option value="both">两种并列</option></select></label>
<label>搜索 <input id="search" type="search" placeholder="父 ID、条件或请求 ID"></label>
<button id="reset" type="button">重置筛选</button><span id="count" aria-live="polite"></span>
</div>
<p class="links"><a href="manifest.json">完整图片清单</a><a href="../evaluation/registration.json">冻结评价登记</a><a href="../evaluation/raw_index.csv">逐输出 RAW 索引</a><a href="AI_PROXY_review_template.csv">AI_PROXY 审阅模板</a></p>
<div id="cards"></div><p class="empty" id="empty" hidden>当前筛选没有请求；所有原始条目仍保留。</p>
<footer id="provenance"></footer>
<noscript>请允许本地 JavaScript 以显示完整索引；所有 PNG 和 manifest.json 均可直接打开，无网络或额外依赖。</noscript>
</main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("data").textContent);
const cards=document.getElementById("cards");
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function metric(v){return typeof v==="number"&&Number.isFinite(v)?(100*v).toFixed(3)+"%":v===null||v===undefined?"NA":String(v);}
for(const q of data.requests){
 const card=el("section",undefined,"card");card.dataset.panel=q.panel;card.dataset.region=q.region_type;
 card.dataset.search=[q.parent_id,q.condition_id,q.request_id,q.tier].join(" ").toLowerCase();
 const h=el("h2");h.append(el("span",q.panel,"badge"));h.append(document.createTextNode(q.request_id));card.append(h);
 card.append(el("p","父 "+q.parent_id+" | 条件 "+q.condition_id,"meta"));
 card.append(el("p","N="+q.N+" | K="+q.K+" | 噪声 n"+q.candidate+" | "+q.region_type+" / "+q.tier,"meta"));
 card.append(el("p","左 → 右："+data.labels.join(" | "),"method-order"));
 const links=el("p",undefined,"links");
 for(const mode of ["colored","neutral"]){
  const a=el("a",mode==="colored"?"打开蓝橙原图":"打开中性原图");a.href=q.images[mode].url;a.target="_blank";a.rel="noopener";links.append(a);
  const figure=el("figure");figure.dataset.mode=mode;
  figure.append(el("figcaption",mode==="colored"?"蓝 C／橙生成 free · 三模型 × 两视角":"中性网格 · 三模型 × 两视角"));
  const img=el("img");img.src=q.images[mode].url;img.loading="lazy";img.decoding="async";
  img.alt=q.request_id+" "+mode+" "+data.labels.join(", ")+" two viewpoints";figure.append(img);card.append(figure);
 }
 card.insertBefore(links,card.querySelector("figure"));
 if(q.scores){
  const detail=el("details");detail.append(el("summary","逐输出客观几何量（不代表语义用途分数；所有比例越低表示对应缺陷越少）"));
  const wrap=el("div",undefined,"score-wrap"),table=el("table"),thead=el("thead"),header=el("tr");
  header.append(el("th","模型"));for(const spec of data.metrics)header.append(el("th",spec[1]));header.append(el("th","I 状态 / 评分状态"));thead.append(header);table.append(thead);
  const tbody=el("tbody");
  for(const label of data.labels){const s=q.scores[label],row=el("tr");row.append(el("td",label));
   for(const spec of data.metrics)row.append(el("td",metric(s.metrics[spec[0]])));
   row.append(el("td",String(s.metrics.ConflictCov_C_status||"NA")+" / "+s.status));tbody.append(row);}
  table.append(tbody);wrap.append(table);detail.append(wrap);
  detail.append(el("p","CF proper 与 CF illegal、FF illegal、I 是不同定义。I 上界保留 UNKNOWN；低支撑或未决状态不能当作精确零。仅列原始客观量，没有新建语义分数。","meta"));card.append(detail);
 }
 cards.append(card);
}
function filter(){
 const panel=document.getElementById("panel").value,region=document.getElementById("region").value,query=document.getElementById("search").value.toLowerCase().trim();
 let visible=0;for(const card of cards.children){card.hidden=!!((panel&&card.dataset.panel!==panel)||(region&&card.dataset.region!==region)||(query&&!card.dataset.search.includes(query)));if(!card.hidden)visible++;}
 document.getElementById("count").textContent="当前 "+visible+" / "+data.requests.length+" 请求（"+visible*3+" / 84 输出）";
 document.getElementById("empty").hidden=visible!==0;
}
for(const id of ["panel","region","search"])document.getElementById(id).addEventListener(id==="search"?"input":"change",filter);
document.getElementById("mode").addEventListener("change",e=>document.body.dataset.view=e.target.value);
document.getElementById("reset").addEventListener("click",()=>{document.getElementById("panel").value="";document.getElementById("region").value="";document.getElementById("search").value="";filter();});
document.getElementById("coverage").textContent="NEWDEV："+data.panel_counts.NEWDEV+" 请求；OLDREG："+data.panel_counts.OLDREG+" 请求。完整56 PNG；同噪声逐模型配对。";
document.getElementById("provenance").textContent="索引登记 SHA256："+data.registration_sha256+" | 图片清单 SHA256："+data.gallery_manifest_sha256+" | 索引脚本 SHA256："+data.builder_sha256+(data.scores_sha256?" | 评分 SHA256："+data.scores_sha256:" | 此页未载入评分");
filter();
</script></body></html>
"""

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def resolve(value):
    text=str(value)
    if os.name!="nt" and re.match(r"^[A-Za-z]:[\\/]",text):
        return Path("/mnt/"+text[0].lower()+"/"+text[3:].replace("\\","/"))
    if os.name=="nt" and re.match(r"^/mnt/[a-z]/",text):
        return Path(text[5].upper()+":/"+text[7:])
    return Path(text)

def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def build(out,include_scores=False,output="index.html"):
    out=resolve(out).resolve();gallery=out/"gallery"
    if Path(output).name!=output or not output.endswith(".html"):raise ValueError("Output must be a local HTML filename")
    reg_path=out/"evaluation/registration.json";manifest_path=gallery/"manifest.json"
    reg=read(reg_path);manifest=read(manifest_path)
    if reg["adapter_sha256"]!=EXPECTED_EVALUATION_SHA:raise ValueError("Unexpected frozen evaluation source")
    if reg["labels"]!=list(LABELS) or len(reg["requests"])!=28:raise ValueError("Expected28 requests and original3labels")
    if manifest["status"]!="COMPLETE" or manifest["PNG_count"]!=56 or manifest["model_outputs"]!=84 or len(manifest["pages"])!=56:
        raise ValueError("All56 originalPNG must be complete before building index")
    registration_sha=digest(reg_path);manifest_sha=digest(manifest_path);page_map={}
    image_hashes={}
    for page in manifest["pages"]:
        identity=page["identity"];key=(identity["request_id"],identity["mode"])
        if key in page_map or identity["columns"]!=list(LABELS):raise ValueError("Duplicate page or wrong model order")
        if identity["evaluation_registration_sha256"]!=registration_sha:raise ValueError("Page registration mismatch")
        if identity["cameras"]!=[[22,-60],[22,120]]:raise ValueError("Unexpected paired camera")
        path=resolve(page["image"]).resolve()
        rel=path.relative_to(gallery).as_posix()
        actual=digest(path)
        if actual!=page["image_sha256"]:raise ValueError("PNG hash changed: "+str(path))
        page_map[key]={"url":quote(rel,safe="/"),"sha256":actual};image_hashes[str(path)]=actual
    expected={(q["request_id"],mode) for q in reg["requests"] for mode in ("colored","neutral")}
    if set(page_map)!=expected:raise ValueError("Gallery misses/adds registered request")
    score_map={};scores_sha=None
    if include_scores:
        scores_path=out/"evaluation/scores_all.json";doc=read(scores_path);scores_sha=digest(scores_path)
        if len(doc["rows"])!=84:raise ValueError("Keep all84 score rows, including failures")
        for row in doc["rows"]:
            key=(row["label"],row["request_id"])
            if key in score_map:raise ValueError("Duplicate scalar row")
            score_map[key]={"status":row["status"],"metrics":row.get("metrics",{})}
        if set(score_map)!={(label,q["request_id"]) for label in LABELS for q in reg["requests"]}:raise ValueError("Scores not exact84 outputs")
    requests=[]
    for q in reg["requests"]:
        item={key:q[key] for key in ("request_id","parent_id","condition_id","panel","candidate","region_type","tier","N","K")}
        item["images"]={mode:page_map[q["request_id"],mode] for mode in ("colored","neutral")}
        if include_scores:item["scores"]={label:score_map[label,q["request_id"]] for label in LABELS}
        requests.append(item)
    payload={"requests":requests,"labels":list(LABELS),"metrics":[list(m) for m in METRICS],
      "panel_counts":{panel:sum(q["panel"]==panel for q in requests) for panel in ("NEWDEV","OLDREG")},
      "registration_sha256":registration_sha,"gallery_manifest_sha256":manifest_sha,"builder_sha256":digest(__file__),"scores_sha256":scores_sha}
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    html=HTML.replace("__PAYLOAD__",encoded);target=gallery/output
    if target.exists() and target.read_text(encoding="utf8")!=html:
        raise ValueError("Refuse overwrite different index; choose --output index_scores.html")
    target.write_text(html,encoding="utf8")
    assert digest(reg_path)==registration_sha and digest(manifest_path)==manifest_sha
    receipt={"status":"COMPLETE","index":str(target),"index_sha256":digest(target),"builder_sha256":digest(__file__),
      "registration_sha256":registration_sha,"gallery_manifest_sha256":manifest_sha,"scores_sha256":scores_sha,
      "requests":28,"model_outputs":84,"PNG_count":56,"images_sha256":image_hashes,"panel_counts":payload["panel_counts"],
      "filtering":"Display only; all28 requests embedded; no output removed","score_included":include_scores,
      "model_calls":0,"scoring_calls":0,"new_renders":0,"source_images_modified":False,"external_dependencies":[]}
    receipt_path=gallery/(output+".receipt.json")
    text=json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+"\n"
    if receipt_path.exists() and receipt_path.read_text(encoding="utf8")!=text:raise ValueError("Existing index receipt differs")
    receipt_path.write_text(text,encoding="utf8")
    return {k:v for k,v in receipt.items() if k!="images_sha256"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--include-scores",action="store_true")
    parser.add_argument("--output",default="index.html")
    args=parser.parse_args()
    print(json.dumps(build(args.out,args.include_scores,args.output),ensure_ascii=False),flush=True)

if __name__=="__main__":main()
