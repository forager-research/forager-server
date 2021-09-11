import React, {useEffect} from "react";
import useResizeObserver from "use-resize-observer/polyfilled";
import LazyLoad, {forceCheck} from "react-lazyload";

import AnnotatedImage from "./AnnotatedImage";

const MARGIN = 3;
const THUMBNAIL_HEIGHT = 200;

const ImageGrid = ({ images, annotations, onClick, minRowHeight, imageAspectRatio, selectedPred }) => {
  const { width, height, ref } = useResizeObserver();
  const handleClick = (e, i) => {
    onClick(e, i);
    e.preventDefault();
  }

  const imagesPerRowFloat = width / (minRowHeight * imageAspectRatio + MARGIN);
  const imagesPerRow = Math.floor(imagesPerRowFloat);
  const imageHeight = Math.floor(minRowHeight * imagesPerRowFloat / imagesPerRow);
  const imageWidth = Math.floor(imageAspectRatio * imageHeight);

  useEffect(() => forceCheck(), [images]);

  return (
    <div className="image-grid" ref={ref}>
      {(width > 0) && images.map((im, i) => {
        return (
          <a href="#" key={i} onClick={(e) => handleClick(e, i)}
            style={{width: imageWidth, marginBottom: MARGIN, marginRight: MARGIN}}
          >
            <div className="image-container">
              <div className={"image " + (selectedPred(i) ? "selected" : "")}>
                <AnnotatedImage
                  url={imageHeight > THUMBNAIL_HEIGHT ? im.src : im.thumb}
                  boxes={(annotations[im.id] && annotations[im.id]["boxes"]) || []}>
                </AnnotatedImage>
              </div>
              <LazyLoad scrollContainer=".modal" height={imageHeight}>
              </LazyLoad>
              {im.distance >= 0 && <div className="image-distance">{im.distance.toFixed(3)}</div>}
            </div>
          </a>);
      })}
    </div>
  );
}

export default ImageGrid;
