import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

class VBallManager:
    def BallTriangulation(self, cap, length = None, YellowThresh = (0.77,130), BackgroundFrames = 50, minContourArea = 20, UpdateMedian = True, MedianUpdateRate = 60, MovementThresh = 20, debug = False, FileName = None):
        SaveAsVideo = type(FileName) == str
        
        def GetBackground(imgs):
            Background = np.median(np.array(imgs), axis=0)
            return Background

        LastNImages = []

        for t in range(BackgroundFrames*MedianUpdateRate):
            res, frame = cap.read()
            if not res:
                return "lenth of vid < background frames * MedianUpdateRate ):<"
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            LastNImages.append(frame)
        Background = GetBackground(LastNImages[::MedianUpdateRate])

        if SaveAsVideo:
            video=cv2.VideoWriter(FileName,
                                  cv2.VideoWriter_fourcc(*'mp4v'), 
                                  cap.get(cv2.CAP_PROP_FPS),
                                  np.array(LastNImages[-1].shape)[[1,0]])

        centers = []
        for t in range(BackgroundFrames*MedianUpdateRate):
            print(f"Frame: {t}", end="\r")

            MovementMask = np.any((np.abs(LastNImages[t] - Background) >= MovementThresh), axis=2)

            ColorMask_ = (0.5*LastNImages[t][:, :, 0] + 0.5*LastNImages[t][:, :, 1] - LastNImages[t][:, :, 2])
            ColorMask = ((ColorMask_-np.min(ColorMask_))/(np.max(ColorMask_)-np.min(ColorMask_))>YellowThresh[0]).astype(np.uint8)

            mask = np.all([MovementMask, ColorMask], axis = 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

            if np.count_nonzero(mask) == 0:
                video.write(cv2.cvtColor((LastNImages[t]/2).astype(np.uint8), cv2.COLOR_RGB2BGR))
                continue
            
            contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if np.max(ColorMask_[mask.astype(np.bool_)]) < YellowThresh[1]:
                video.write(cv2.cvtColor((LastNImages[t]/2).astype(np.uint8), cv2.COLOR_RGB2BGR))
                continue
            
            contour = max(contour, key=cv2.contourArea)[:, 0]
            if cv2.contourArea(contour) < minContourArea:
                video.write(cv2.cvtColor((LastNImages[t]/2).astype(np.uint8), cv2.COLOR_RGB2BGR))
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x+w/2), int(y+h/2))

            if SaveAsVideo:
                video.write(cv2.circle(cv2.cvtColor((LastNImages[t]/2).astype(np.uint8), cv2.COLOR_RGB2BGR), center, 5, (0, 0, 255), -1))
            
            if debug:
                print("\n\n\n")
                plt.imshow(LastNImages[t])
                plt.scatter(*center, color="red", s=1)
                plt.show()
                # plt.imshow(Background/255)
                # plt.show()
                # plt.imshow(MovementMask1)
                # plt.show()
                # plt.imshow(ColorMask)
                # plt.show()
                plt.imshow(mask)
                plt.scatter(*center, color="red", s=1)
                plt.show()

        if not UpdateMedian: 
            del(LastNImages)
        else: 
            LastNImages = LastNImages[::MedianUpdateRate]

        while True:
            print(f"Frame: {t}", end="\r")

            if length: 
                if t >= length: break
            t += 1

            res, frame = cap.read()
            if not res:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if UpdateMedian:
                if t % MedianUpdateRate == 0:
                    LastNImages = LastNImages[1:] + [frame]
                    Background = GetBackground(LastNImages)

            MovementMask = np.sum((np.abs(frame - Background) >= MovementThresh), axis=2).astype(np.bool_)

            ColorMask_ = (0.5*frame[:, :, 0] + 0.5*frame[:, :, 1] - frame[:, :, 2])
            ColorMask = ((ColorMask_-np.min(ColorMask_))/(np.max(ColorMask_)-np.min(ColorMask_))>YellowThresh[0]).astype(np.uint8)

            mask = np.all([MovementMask, ColorMask], axis = 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

            if np.count_nonzero(mask) == 0:
                video.write(cv2.cvtColor((frame/2).astype(np.uint8), cv2.COLOR_RGB2BGR))
                continue
            
            if np.max(ColorMask_[mask.astype(np.bool_)]) < YellowThresh[1]:
                video.write(cv2.cvtColor((frame/2).astype(np.uint8), cv2.COLOR_RGB2BGR))
                continue

            contour, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            contour = max(contour, key=cv2.contourArea)[:, 0]
            if cv2.contourArea(contour) < minContourArea:
                video.write(cv2.cvtColor((frame/2).astype(np.uint8), cv2.COLOR_RGB2BGR))
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x+w/2), int(y+h/2))
            centers.append(center)

            if SaveAsVideo:
                video.write(cv2.circle(cv2.cvtColor((frame/2).astype(np.uint8), cv2.COLOR_RGB2BGR), center, 5, (0, 0, 255), -1))

            if debug:
                plt.imshow(frame)
                plt.scatter(*center, color="red", s=1)
                plt.show()
                # plt.imshow(Background/255, aspect='auto')
                # plt.show()
                # plt.imshow(MovementMask)
                # plt.show()
                # plt.imshow(ColorMask)
                # plt.show()
                plt.imshow(mask)
                plt.scatter(*center, color="red", s=1)
                plt.show()
        centers = np.array(centers)
        
        if SaveAsVideo:
            video.release()

            return centers, video

        return centers
    
    def ProcessVideo(self, vid, StartFrame, StorageFile=False, VidStorageFile=False, length=False):
        cap = cv2.VideoCapture(vid)
        cap.set(cv2.CAP_PROP_POS_FRAMES, StartFrame-1)
        
        centers, _ = self.BallTriangulation(cap, length=length, FileName=VidStorageFile)

        if StorageFile:
            file = open(StorageFile,"wb")
            pickle.dump(centers, file)
            file.close()