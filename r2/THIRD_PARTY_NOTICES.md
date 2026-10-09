# Third-party sources

This research extension is not an official method or release of the original MeshFlow authors.

The necessary MeshFlow backbone, Native T1 and C-only Geo implementations originate from
WANG-Ruipeng/MeshFlow, source commit ae91eaa6f882a7d311233fa32053b7cd5de7ebe0,
itself based on qiisun/MeshFlow. The original MIT license and copyright are retained in LICENSE
and source headers. This repository-level MIT notice is not an assertion that all vendored files share those terms.

The migrated `src/meshflow_control/models/backbone/utils.py` preserves the NVIDIA 2024 notice at lines 1-10. It states that use, reproduction, disclosure or distribution requires an express license agreement from NVIDIA. The independent permission or license basis for that retained notice has not been verified in this audit. Retaining it records its provenance; it does not establish additional rights. No experimental code or local execution behavior was changed in response to this documentation finding.
Training coupling and geometric scoring portions are adapted from preserved local research source
in the same fork; exact source files and hashes are recorded in docs/source_map.md and component maps.
Only necessary components are migrated. Historical models, data and artifacts are not vendored.

Runtime dependencies retain their own licenses. Weights and datasets must be provided separately;
their distribution is not authorized by this code license.
