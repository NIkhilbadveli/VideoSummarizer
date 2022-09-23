import os
from utils import FrameExtractor
from text_summarizer import TextSummarizer
from image_captioner import ImageCaptioner

video_input = 'Input Videos/shorts.mp4'

# Extract ExtractedFrames and save them in the following folder
frames_folder = 'ExtractedFrames/' + os.path.normpath(video_input).split(os.sep)[-1].split('.')[0]
FrameExtractor().extract_frames(video_input, frames_folder, n=30)

# Now, get a list of the images inside this folder
all_frames = [frames_folder + '/' + file_name for file_name in os.listdir(frames_folder)]

# Now, pass this to the captioner to get a list of captions
ic = ImageCaptioner()
all_captions = []
for caption in ic.generate_captions(all_frames):
    if caption not in all_captions:
        all_captions.append(caption)

total_captioned_text = '. '.join(all_captions)  # Take only the first 1024 tokens due to limitation of BART summarizer.
# print('\nHere is the total captioned text of the video:-\n')
# print(total_captioned_text)

# Finally, join all the captions by ' ' and pass it to the summarizer
ts = TextSummarizer(max_length=250)
video_summary = ts.summarize(total_captioned_text)

print('\nHere is the summary of the video:-\n')
print(video_summary)
