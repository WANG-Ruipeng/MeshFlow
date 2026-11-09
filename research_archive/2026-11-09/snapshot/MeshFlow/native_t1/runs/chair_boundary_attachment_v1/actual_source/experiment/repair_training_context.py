"""Before first model call, retain original backward runtime context."""
from pathlib import Path
p=Path(__file__).with_name('training.py');s=p.read_text()
s=s.replace("        _,bs=call_sync(budget,'training_backward',loss.backward,context)","        with native_execution(model,autocast=True):\n            _,bs=call_sync(budget,'training_backward',loss.backward,context)")
s=s.replace("step=int(meta['attachment_step']);stream=AttachmentStream(SOURCE,CHAIR,out,arm)","step=int(meta['attachment_step']);stream=AttachmentStream(SOURCE,CHAIR,out,arm)")
s=s.replace("tick=time.perf_counter();samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick", "committed_stream_state=stream.state_dict()\n                tick=time.perf_counter();samples=stream.next_effective_batch();input_seconds=time.perf_counter()-tick")
s=s.replace("        if not in_update and step>saved_step:\n            try:persist()", "        if not in_update and stream.batch_index!=step:\n            stream.load_state_dict(committed_stream_state)\n        if not in_update and step>saved_step:\n            try:persist()")
p.write_text(s)
print('Backward context and uncalled-input cursor handling ready')
