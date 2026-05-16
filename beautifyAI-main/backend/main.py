import argparse
import sys
import os
import subprocess
import shutil
import uuid

def run_beautify_model(input_path, output_path, intensity="0.5"):
    print("Initializing Official CodeFormer High-End Face Beautification...")
    
    codeformer_dir = os.path.join(os.path.dirname(__file__), 'CodeFormer-master')
    temp_out_dir = os.path.join(os.path.dirname(__file__), 'temp_cf_out')
    
    if not os.path.exists(codeformer_dir):
        print("Error: CodeFormer directory not found.")
        sys.exit(1)
        
    # Ensure absolute paths
    input_abs = os.path.abspath(input_path)
    out_dir_abs = os.path.abspath(temp_out_dir)
    
    # Create temp dir
    os.makedirs(out_dir_abs, exist_ok=True)
    
    # Workaround: CodeFormer strictly checks if input ends with .jpg/.png
    # But multer often uploads without an extension.
    # We will copy it to a temporary .jpg file.
    temp_input_jpg = os.path.join(out_dir_abs, f"input_{uuid.uuid4().hex}.jpg")
    shutil.copyfile(input_abs, temp_input_jpg)
    
    try:
        # The user's UI slider goes from 0.0 (No Enhancement) to 1.0 (Max Enhancement).
        # CodeFormer's -w is Fidelity: 1.0 (Exact original, no enhancement) to 0.0 (Max Quality/Enhancement).
        # We invert the intensity so the slider acts as an "Enhancement" slider.
        fidelity_weight = max(0.0, min(1.0, 1.0 - float(intensity)))
        
        # Run the official CodeFormer script
        cmd = [
            sys.executable, "inference_codeformer.py",
            "-w", str(fidelity_weight),
            "-i", temp_input_jpg,
            "-o", out_dir_abs
        ]
        
        # Set PYTHONPATH so basicsr/facelib can always be found (CodeFormer-master is the source root)
        env = os.environ.copy()
        existing_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = codeformer_dir + (os.pathsep + existing_pypath if existing_pypath else "")
        
        result = subprocess.run(cmd, cwd=codeformer_dir, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"CodeFormer failed:\n{result.stderr}")
            sys.exit(1)
            
        # CodeFormer outputs to <out_dir>/final_results/<basename>.png
        basename = os.path.splitext(os.path.basename(temp_input_jpg))[0]
        result_img = os.path.join(out_dir_abs, "final_results", f"{basename}.png")
        
        if os.path.exists(result_img):
            # Move it to the exact output path requested by Node.js
            shutil.move(result_img, output_path)
            print(f"Successfully beautified using Official CodeFormer. Saved to {output_path}")
        else:
            print("Error: CodeFormer did not generate an output image.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary CodeFormer output directory
        if os.path.exists(out_dir_abs):
            shutil.rmtree(out_dir_abs, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Face Beautification Script")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output image path")
    parser.add_argument("-int", "--intensity", default="0.5", help="Intensity/Fidelity weight")
    
    args = parser.parse_args()
    
    run_beautify_model(args.input, args.output, args.intensity)
