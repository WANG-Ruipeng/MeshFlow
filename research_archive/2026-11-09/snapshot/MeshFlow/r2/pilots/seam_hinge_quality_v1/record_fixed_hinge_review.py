"""Actual inspection of both registered fixed spatial HINGE windows."""
from .evaluation import C,OUT

def run():
    records=[]
    for rid,note in [
        ('608b46b0_n3','原S2000红/橙两段之间的间隔在015/025后仍在；CLEAN在同一front窗口下的两段灰线与间隔也未被两档填满。SMOOTH三列窗口没有灰线，但完整图已显示下层移到该窄Y窗口之外，不能称为悬片消失。两新模型不沿用原77/113槽的语义身份。'),
        ('3f741759_n2','原S2000的上方红色短线在015/025后保留；它的完整索引图存在非流形边联系，不能因合法图单例就称为悬片。CLEAN座下宽缺口在两档后仍清楚；SMOOTH上方短线及座下完整框也都保留。没有从本窗口观察到两档解决这些关注位置。')]:
        p=OUT/'sections/fixed_cases'/(rid+'_hinge.png')
        records.append(dict(request_id=rid,image_path=str(p),image_sha256=C.sha(p),observation=note,
                            reviewer='AI_PROXY',blind=False,actually_viewed=True))
    result=dict(status='ACTUALLY_VIEWED_TWO_FIXED_WINDOWS',rows=records,
        scope='Identical historical spatial planes and bounds; three models x POST/HINGE015/HINGE025. No new model/scorer calls.',
        limit='Finite section gaps are not complete mesh connectivity certificates; no semantic correspondence of slots across trained models.')
    C.write(OUT/'sections/fixed_cases/actual_hinge_review.json',result);return result

if __name__=='__main__':print(run()['status'],flush=True)
