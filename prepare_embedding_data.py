import os
import shutil


root = os.getcwd()
original_data_path = os.path.join(root, 'LibriSpeech/test-clean')
final_dir = os.path.join(root, 'final_emb_data')
if not os.path.exists(final_dir): os.mkdir(final_dir)

speaker_list = [1089, 121, 1284, 1580, 2094, 237, 2830, 3570, 3729, 4446]

for speaker_id in os.listdir(original_data_path):
    if int(speaker_id) in speaker_list:
        for uttid in os.listdir(os.path.join(original_data_path, speaker_id)):
            for file in os.listdir(os.path.join(original_data_path, speaker_id, uttid)):
                if file.endswith('.flac'):
                    src = os.path.join(original_data_path, speaker_id, uttid, file)
                    # if not os.path.exists(os.path.join(final_dir, speaker_id)): 
                    #     os.mkdir(os.path.join(final_dir, speaker_id))
                    dst = os.path.join(final_dir, file)
                    shutil.copy(src, dst)



# for speaker_id in os.listdir(original_data_path):

#     for uttid in os.listdir(os.path.join(original_data_path, speaker_id)):
#         for file in os.listdir(os.path.join(original_data_path, speaker_id, uttid)):
#             if file.endswith('.flac'):
#                 src = os.path.join(original_data_path, speaker_id, uttid, file)
#                 # if not os.path.exists(os.path.join(final_dir, speaker_id)): 
#                 #     os.mkdir(os.path.join(final_dir, speaker_id))
#                 dst = os.path.join(final_dir, file)
#                 shutil.copy(src, dst)