import os
from utils import FrameExtractor
from text_summarizer import TextSummarizer
from image_captioner import ImageCaptioner
import streamlit as st
import sys
from threading import current_thread
from contextlib import contextmanager
from io import StringIO


@st.cache(allow_output_mutation=True)
def load_models():
    # Initialize the models
    ic = ImageCaptioner()
    ts = TextSummarizer(max_length=250)
    return ic, ts


def generate_summary(video_in, ic, ts):
    """Generates a summary of a given video"""
    # Extract ExtractedFrames and save them in the following folder
    frames_folder = 'ExtractedFrames/' + os.path.normpath(video_in).split(os.sep)[-1].split('.')[0]
    FrameExtractor().extract_frames(video_in, frames_folder, n=10, max_num_frames=180)

    # Now, get a list of the images inside this folder
    all_frames = [frames_folder + '/' + file_name for file_name in os.listdir(frames_folder)]

    # Now, pass this to the captioner to get a list of captions
    all_captions = []
    for caption in ic.generate_captions(all_frames):
        if caption not in all_captions:
            all_captions.append(caption)

    total_captioned_text = '. '.join(
        all_captions)  # Take only the first 1024 tokens due to limitation of BART summarizer.
    # print('\nHere is the total captioned text of the video:-\n')
    # print(total_captioned_text)

    # Finally, join all the captions by ' ' and pass it to the summarizer
    video_summary = ts.summarize(total_captioned_text)

    # print('\nHere is the summary of the video:-\n')
    # print(video_summary)

    return video_summary


# @contextmanager
# def st_redirect(src, dst):
#     placeholder = st.empty()
#     output_func = getattr(placeholder, dst)
#
#     with StringIO() as buffer:
#         old_write = src.write
#
#         def new_write(b):
#             if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
#                 buffer.write(b + '')
#                 output_func(buffer.getvalue() + '')
#             else:
#                 old_write(b)
#
#         try:
#             src.write = new_write
#             yield
#         finally:
#             src.write = old_write
#
#
# @contextmanager
# def st_stdout(dst):
#     "this will show the prints"
#     with st_redirect(sys.stdout, dst):
#         yield


def main():
    st.title('Video Summarizer')
    st.write(
        'Generate a summary of the uploaded video (only takes the first 1 min). This app uses two separate models and '
        'it can take sometime to download them for the first time. Subsequent '
        'uses should be much faster.')

    # Load the models first
    img_cpnr, txt_smz = load_models()

    # Get the video file uploaded
    uploaded_video = st.file_uploader('Upload a short video', type=['mp4', 'mpeg'])

    if uploaded_video is not None:
        video_file = uploaded_video.name
        with open(video_file, 'wb') as f:
            f.write(uploaded_video.read())

        # st.markdown(f"""
        #     ### Files
        #     - {video_file}
        #     """, unsafe_allow_html=True)  # display file name

        lc, rc = st.columns(2)
        # Use lc to show the uploaded video

        # Show a generate button
        gen_btn = st.button('Generate Summary')

        if gen_btn:
            st.write('Generating summary now...')
            summary = generate_summary(video_file, ic=img_cpnr, ts=txt_smz)
            st.write('Generating summary now... Done')
            rc.title('Summary generated')
            rc.write(summary)


if __name__ == '__main__':
    main()
