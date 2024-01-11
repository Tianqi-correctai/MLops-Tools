import streamlit_antd_components as sac
import streamlit as st
from pathlib import Path 

# show things under backends/data 
path = Path("../backends/data")
train_path = path / "train"
val_path = path / "val"
infer_path = path / "inference"
sources_path = path / "sources"
export_path = path / "export"

TEXT_FORMATS = [".txt", ".json", ".xml", ".csv", ".yaml", ".yml"]
IMG_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"]
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".mpeg"]
LOG_FORMATS = [".log"]
MODEL_FORMATS = [".pt", ".pth", ".h5", ".hdf5", ".ckpt", ".pb", ".onnx", ".engine"]

indices = {}
clickable = []
def build_tree_folder(path, index=0, level=0, max_level=2):
    children = []
    level += 1
    if not path.exists():
        return sac.TreeItem(str(path.resolve()), icon="folder-x", children=[])
    
    for file in sorted(path.iterdir(), reverse=True):
        index += 1
        indices[str(file.resolve())] = index

        if file.is_dir() and level < max_level:
            children.append(build_tree_folder(file, index, level, max_level))
            continue
        
        clickable.append(str(file.resolve()))

        if file.suffix.lower() in TEXT_FORMATS:
            children.append(sac.TreeItem(str(file.resolve()), icon="card-text"))
        elif file.suffix.lower() in LOG_FORMATS:
            if file.name == "meta.log":
                children.append(sac.TreeItem(str(file.resolve()), icon="info-square-fill"))
            else:
                children.append(sac.TreeItem(str(file.resolve()), icon="terminal-fill"))
        elif file.suffix.lower() in MODEL_FORMATS:
            children.append(sac.TreeItem(str(file.resolve()), icon="stars"))

        elif file.suffix.lower() in IMG_FORMATS:
            children.append(sac.TreeItem(str(file.resolve()), icon="image"))
        elif file.suffix.lower() in VIDEO_FORMATS:
            children.append(sac.TreeItem(str(file.resolve()), icon="play-btn-fill"))
        else:
            children.append(sac.TreeItem(str(file.resolve()), icon="collection"))

    return sac.TreeItem(str(path.resolve()), icon="folder", children=children)

def text_reader(file):
    data = file.read_text()
    if len(data) > 10000:
        data = data[:10000]
        data += "\n\n... truncated ..."
    return data


def st_preview(file):
    if file.suffix.lower() in TEXT_FORMATS:
        st.code(text_reader(file))
    elif file.suffix.lower() in LOG_FORMATS:
        st.code(text_reader(file))
    elif file.suffix.lower() in IMG_FORMATS:
        st.image(str(file.resolve()))
    elif file.suffix.lower() in VIDEO_FORMATS:
        st.video(str(file.resolve()))
    else:
        st.write("No preview available.")


st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([0.6,0.6,1.6])
with col1:
    st.subheader("Task Explorer")
    selected = sac.tree(items=[
        build_tree_folder(train_path),
        build_tree_folder(val_path),
        build_tree_folder(infer_path),
        build_tree_folder(sources_path),
        build_tree_folder(export_path)
    ], format_func=lambda x:Path(x).name, icon='table', open_all=True)

with col2:
    st.subheader("Task Folder")
    if len(selected) > 0 and selected[0] in clickable:
        if Path(selected[0]).is_dir():
            selected2 = sac.tree(items=[
                build_tree_folder(Path(selected[0]), max_level=10)
            ], format_func=lambda x:Path(x).name, icon='table', open_all=True)
            with col3:
                
                if len(selected2) > 0 and selected2[0] in clickable:
                    col3_1, col3_2, col3_3 = st.columns([1, 2, 1])
                    with col3_1:
                        st.subheader("Preview")
                    with col3_3:
                        with open(selected2[0], 'rb') as f:
                            st.download_button("Download", f, Path(selected2[0]).name, use_container_width=True)
                    file = Path(selected2[0])
                    st_preview(file)
        else:
            with col3:
                col3_1, col3_2, col3_3 = st.columns([1, 2, 1])
                with col3_1:
                    st.subheader("Preview")
                with col3_3:
                    with open(selected[0], 'rb') as f:
                        st.download_button("Download", f, Path(selected[0]).name, use_container_width=True)
                file = Path(selected[0])
                st_preview(file)



