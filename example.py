from VisionTracker.VisionTracker_object import VisualTracker

if __name__ == "__main__":
    try:
        mytracker = VisualTracker()
        mytracker.track()
    except Exception as e:
        print(e)
