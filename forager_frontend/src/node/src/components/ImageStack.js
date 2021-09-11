import React from "react";
import times from "lodash/times";

const ImageStack = ({ id, onClick, images, showLabel, labelText, showDistance, distanceText }) => {
  const shouldShowLabel = showLabel || !!(labelText);
  const shouldShowDistance = showDistance;
  return (
    <a className={`stack ${shouldShowLabel ? "" : "nolabel"}`} onClick={onClick}>
      {times(Math.min(4, images.length), (i) =>
        <div key={`stack-${i}`} className="thumb-container">
          <img className="thumb" loading="lazy" src={images[i].thumb}></img>
          {shouldShowDistance && i == 0 && <div className="image-distance">{distanceText.toFixed(3)}</div>}
        </div>
      )}
      {shouldShowLabel && <div className="label">
        {labelText || <>
          <b>Cluster {id + 1}</b> ({images.length} image{images.length !== 1 && "s"})
        </>}
      </div>}
    </a>
  );
}

export default ImageStack;
