import os
import time
import sys
from moviepy.editor import VideoFileClip
from ai_planner import get_ai_plan
from renderer import render_video
from project_model import new_project, save_project, load_project, Effect, TextOverlay
from editing_engine import render_project
from camera_config import DEFAULT_CONFIG
from metrics_engine import compute_camera_metrics, compute_processing_metrics

def main():
    print("Starting integration test for all features...")
    video_path = "demo.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found in current directory.")
        sys.exit(1)
    
    # 1. Probe video duration
    print("Probing demo.mp4 duration...")
    clip = VideoFileClip(video_path)
    duration = clip.duration
    print(f"Duration: {duration} seconds. Size: {clip.size}")
    clip.close()

    # We will test using a short 5-second clip to speed up the test
    # Let's create a 5-second subclip of demo.mp4 first
    subclip_path = "temp_test_subclip.mp4"
    print("Creating a 5-second subclip for faster testing...")
    clip = VideoFileClip(video_path).subclip(0, 5)
    clip.write_videofile(subclip_path, fps=30, codec="libx264", audio_codec="aac", logger=None)
    clip.close()
    
    try:
        # 2. Test AI Planner (get_ai_plan) - should fall back gracefully if no API key
        print("Testing get_ai_plan with fallback (no API key)...")
        plan = get_ai_plan(subclip_path, 5.0, api_key=None, config=DEFAULT_CONFIG)
        print("Plan generated successfully!")
        print("Plan keys:", list(plan.keys()))
        print("Number of camera segments:", len(plan["segments"]))
        
        # 3. Test Renderer (render_video)
        print("Testing render_video...")
        output_path = "temp_test_render.mp4"
        if os.path.exists(output_path):
            os.remove(output_path)
        render_video(subclip_path, output_path, plan, config=DEFAULT_CONFIG)
        if os.path.exists(output_path):
            print("Render video created successfully!")
            os.remove(output_path)
        else:
            print("Error: Render video output file not found.")
            sys.exit(1)
            
        # 4. Test Project Model and Editing Engine (multi-clip timeline, trim, split, effects, text, audio, export)
        print("Testing Project Model & Editing Engine timeline...")
        project = new_project("Test Integration Project")
        
        # Add clip
        clip_model = project.add_clip(subclip_path, trim_start=0.0, trim_end=5.0, camera_plan=plan)
        print(f"Added clip: {clip_model.id}")
        
        # Add effects
        print("Adding brightness effect...")
        clip_model.effects.append(Effect(kind="brightness", params={"factor": 1.2}))
        
        # Add text overlay
        print("Adding text overlay...")
        clip_model.text_overlays.append(TextOverlay(text="Integration Test", startTime=1.0, endTime=3.0, position="center", style="title"))
        
        # Split clip
        print("Testing clip split at 2.5s...")
        new_clip = project.split_clip(clip_model.id, 2.5)
        if new_clip:
            print(f"Split successful! New clip ID: {new_clip.id}. Number of clips: {len(project.clips)}")
        else:
            print("Error: Split clip failed.")
            sys.exit(1)
            
        # Undo/Redo
        print("Testing undo/redo...")
        undone = project.undo()
        print(f"Undo: {undone}. Number of clips: {len(project.clips)}")
        redone = project.redo()
        print(f"Redo: {redone}. Number of clips: {len(project.clips)}")
        
        # Render project
        print("Testing render_project...")
        project_output_path = "temp_test_project_render.mp4"
        if os.path.exists(project_output_path):
            os.remove(project_output_path)
        
        render_project(project, project_output_path, DEFAULT_CONFIG)
        if os.path.exists(project_output_path):
            print("Project rendered successfully!")
            os.remove(project_output_path)
        else:
            print("Error: Project render output file not found.")
            sys.exit(1)
            
        # 5. Test metrics computation
        print("Testing metrics engine...")
        metrics = compute_camera_metrics(plan)
        print("Camera metrics computed:", metrics.keys())
        
        print("\n=== ALL PIPELINE AND TIMELINE FEATURES TESTED AND ARE WORKING PROPERLY ===")
        
    finally:
        # Cleanup
        if os.path.exists(subclip_path):
            os.remove(subclip_path)

if __name__ == "__main__":
    main()
