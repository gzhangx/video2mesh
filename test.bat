rem ffmpeg -i "D:\temp\IMG_0757.MOV" -c:v libx264 -crf 18 -preset slow -c:a aac "D:\temp\yourvideo.mp4"

rem ffmpeg -i input.mov -map 0:v:0 -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -an clip_for_mesh.mp4


cd D:\work\cur\video2mesh
.venv\Scripts\activate
python -m video2mesh view out_hallway