from pathlib import Path
p=Path('/mnt/e/MeshFlow-Control/src/meshflow_control/evaluation/render.py')
s=p.read_text().replace('from mpl_toolkits.mplot3d.art3d import Poly3DCollection','from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection')
start=s.index('            for tri,col in ((F')
end=s.index('            cc=center',start)
s=s[:start]+'''            tri=np.concatenate([F,C]);colors=['#e7994c']*len(F)+['#3a82c5']*len(C)
            displayed=np.asarray(tri)[:,:,(0,2,1)]
            if view=='wire':
                edges=displayed[:,((0,1),(1,2),(2,0)),:].reshape(-1,2,3)
                artist=Line3DCollection(edges,colors=np.repeat(colors,3),linewidths=.38)
            else:
                artist=Poly3DCollection(displayed,facecolors=colors,edgecolors='#444444',linewidths=.10,alpha=1)
            ax.add_collection3d(artist)
'''+s[end:]
p.write_text(s)
