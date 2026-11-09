# Teacher-to-student tail geometry diagnosis

Fixed 8 pairs x 9 lambdas = 72 diagnostic geometries. No training, network, optimization, MC, repair or output selection.
Reference: the same frozen radial field. No new exact intersection/topology audit; angle gates and positive direction determinants do not certify full validity.

|UID / target|global direction RMSE deg|teacher-tail / global coordinate MSE|student min margin deg|student p05 margin deg|teacher tail retained /26|min face teacher->student|
|---|---:|---:|---:|---:|---:|---|
|7f1be3e7 / old|0.442887|0.593|1.106597|1.332672|24|371 -> 363|
|7f1be3e7 / margin|0.476566|0.851|1.726461|0.890152|21|135 -> 373|
|2379915c / old|0.169303|1.076|-0.263871|-0.098375|24|384 -> 384|
|2379915c / margin|0.268947|1.207|-0.276827|0.320662|24|132 -> 384|
|8c919e99 / old|0.043872|0.863|-0.046446|-0.040201|26|128 -> 128|
|8c919e99 / margin|0.220863|1.206|2.705824|-0.286813|24|404 -> 38|
|736820a8 / old|0.002893|1.278|0.016937|-0.007723|26|139 -> 139|
|736820a8 / margin|0.207324|1.608|0.512854|-0.071287|24|353 -> 353|

The global MSE averages all 258 vertex coordinates. Minimum angle depends on one local triangle; p05 depends on two order statistics and can change its supporting faces. Local error, differential corner motion, changing radii and rank migration are not bounded by a small global average.
Tail error enrichment is descriptive and is not proof that a particular vertex caused failure. A ratio below one does not guarantee quality: small errors can exceed a small margin or change the ordering.
Each endpoint is re-queried and checked against the saved directions, radii, vertices and angle/area metrics. Endpoints are not recomputed by a model. Interpolants are not deployable replacements and no lambda is selected.
