import cv2

video = cv2.VideoCapture('video.mp4')
car_cascade = cv2.CascadeClassifier('cars.xml')

# Get the video dimensions 
frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Adjust window size
windowWidth = 1200
windowHeight = int(windowWidth * (frameHeight / frameWidth))

try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video.")
            break

        frame = cv2.resize(frame, (windowWidth, windowHeight))  # Resize frame to fit the window size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        cars = car_cascade.detectMultiScale(gray, 1.3, 2, minSize=(30, 30))
        carCount = len(cars)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        cv2.putText(frame, f"Cars detected in frame: {carCount}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Output", frame)

        # Exit with esc key
        if cv2.waitKey(33) == 27:
            break

except Exception as e:
    print("Error:", e)
finally:
    print("Cleaning...")
    video.release()
    cv2.destroyAllWindows()
