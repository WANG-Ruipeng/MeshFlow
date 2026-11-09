# Actual scoring source and asset provenance

The scorers/ tree contains the 19 exact small sources named in the saved calibration
POST identity.definition_hashes. All 256 saved POST identities have the same set.
Source and copied bytes were checked against those hashes; no scorer/model ran.

scorers_manifest.json and .csv map original absolute/project-relative paths to
copied paths and hashes. The external MeshFlow-Control tree remains distinct.

protected_assets_manifest.json freshly verifies small conditions, surface queries,
region metadata and frozen control records. FULL_S6000 and official D0 hashes
are reused from frozen bindings and completed execution evidence; only existence
and size are checked here, without re-reading the large weights.

frozen_request_noise_hashes.json records 64 calibration, 44 final and 512 training
request identities. Files were checked for presence; hashes are frozen-manifest
values, explicitly not a new tensor rehash.

This is packaging preparation only: no final ZIP, weight copy, model switch,
source edit, scoring, training or GPU calls.
