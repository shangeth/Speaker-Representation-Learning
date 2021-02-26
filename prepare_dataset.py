import os
import shutil


root = os.getcwd()

original_data_path = os.path.join(root, 'LibriSpeech/train-clean-360')
final_dir = os.path.join(root, 'final_repr_data')
if not os.path.exists(final_dir): os.mkdir(final_dir)


for speaker_id in os.listdir(original_data_path):
    for uttid in os.listdir(os.path.join(original_data_path, speaker_id)):
        for file in os.listdir(os.path.join(original_data_path, speaker_id, uttid)):
            if file.endswith('.flac'):
                src = os.path.join(original_data_path, speaker_id, uttid, file)
                if not os.path.exists(os.path.join(final_dir, speaker_id)): 
                    os.mkdir(os.path.join(final_dir, speaker_id))
                dst = os.path.join(final_dir, speaker_id, file)
                shutil.copy(src, dst)


# OPTIONAL, if you want to also add Dev set to train
# original_data_path = os.path.join(root, 'LibriSpeech/dev-clean')
# final_dir = os.path.join(root, 'final_repr_data')
# if not os.path.exists(final_dir): os.mkdir(final_dir)


# for speaker_id in os.listdir(original_data_path):
#     for uttid in os.listdir(os.path.join(original_data_path, speaker_id)):
#         for file in os.listdir(os.path.join(original_data_path, speaker_id, uttid)):
#             if file.endswith('.flac'):
#                 src = os.path.join(original_data_path, speaker_id, uttid, file)
#                 if not os.path.exists(os.path.join(final_dir, speaker_id)): 
#                     os.mkdir(os.path.join(final_dir, speaker_id))
#                 dst = os.path.join(final_dir, speaker_id, file)
#                 shutil.copy(src, dst)