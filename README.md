# 3D-Calibration-Tool

- Change the source, target, and save pathes of 3D models in calibrate.py.

- > python calibrate.py

- Mark the selected feature points in the first UI frame: press P to pick a point, D to delete the last point, S to save the current screenshot, Q to quit.

- The program will display the marked points in the saved screenshot and generate another UI frame for the user to mark corresponding feature points. Done with pressing Q.

- The optimal scale, rotation, and translation will be calculated based on the marked pairs and transform the source model to align with the target. Result will be saved in sv_path.