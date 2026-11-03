# Video assets (1920x1080, 30 fps)

Made from the 3D viewer (`scripts/record_video.py`, `scripts/record_mesh.py`). "Hand-finished" models: the pipeline's
build, then mechanisms placed by hand with the project's own builders and every one re-checked by the checker - say
so on screen ("finished by hand") when they are shown.

## clips/ - the models moving (cut one after another for the montage)
| file | what | use |
|---|---|---|
| 01_Camry.mp4 | red Camry orbiting, doors swinging, wheels turning | after the "photo -> LEGO" cut |
| 02_Camry_build.mp4 | the video-game build: chassis first (Technic bricks, axles through their holes, wheels), body bottom-up, speeding up until it is one piece | "every step visualized": the build |
| 03_Supercar.mp4 | blue supercar (photo glass, doors) | montage |
| 04_747.mp4 | 747: 4 fans spinning in hollow nacelles, flap + aileron each wing, gear | montage |
| 05_Helicopter.mp4 | helicopter: main rotor (turntable + blades) and tail rotor spinning | montage |
| 06_Cessna.mp4 | Cessna: propeller spinning, control surfaces moving, on its gear | montage |
| montage.mp4 | 03-06 cut back to back (the last ~30 s) | end |

## steps/ - "every step visualized" (Camry)
| file | step |
|---|---|
| 1_photo.png | the photo |
| 2_sam31_mask.png | SAM 3.1 finds the car |
| 3_sam31_glass.png, overlay_windshield.png, overlay_car_windows.png | SAM 3.1 finds the glass Spark named ("windshield", "side windows") |
| 3b_glass_marked_photo.png | the glass marked on the photo for SAM 3D |
| 4_sam3d_mesh.mp4 | SAM 3D's mesh turning, the windshield marked |
| 5_voxels.png | the mesh as LEGO-sized cells (front / side / top) |
| 6_spark_layout_side.png | Spark points at the wheels and doors |
| (docs/feedback-example.md) | Glimmer's answer, the checker's sentence, the fix - put the text on screen |
| clips/02_Camry_build.mp4, clips/01_Camry.mp4 | the build, then it works |
