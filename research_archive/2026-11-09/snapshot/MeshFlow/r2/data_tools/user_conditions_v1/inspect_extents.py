from prepare import DEFAULT_OUT,read
for r in sorted(read(DEFAULT_OUT/'conditions.json')['rows'],key=lambda r:r['region_area_fraction']):
 if r['region_area_fraction']<.15 or r['region_area_fraction']>.8 and r['tier']!='full':
  print(r['condition_id'],r['K'],'C/R',round(r['region_area_fraction'],4),'C/M',round(r['parent_area_fraction'],4),r['requested_tiers'])
