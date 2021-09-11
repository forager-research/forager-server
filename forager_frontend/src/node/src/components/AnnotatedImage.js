import React, {useEffect, useState, useRef, useLayoutEffect} from "react";
import {Stage, Layer, Image, Line, Text} from "react-konva";
import Konva from "konva";
import useImage from "use-image";


const AnnotatedImage = ({ url, boxes, onClick }) => {
  const targetRef = useRef();
  const [containerWidth, setContainerWidth] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);

  // holds the timer for setTimeout and clearInterval
  let movement_timer = null;

  // the number of ms the window size must stay the same size before the
  // dimension state variable is reset
  const RESET_TIMEOUT = 100;

  const test_dimensions = () => {
    // For some reason targetRef.current.getBoundingClientRect was not available
    // I found this worked for me, but unfortunately I can't find the
    // documentation to explain this experience
    if (targetRef.current) {
      setContainerWidth(targetRef.current.offsetWidth);
      setContainerHeight(targetRef.current.offsetHeight);
    }
  }

  // This sets the dimensions on the first render
  useLayoutEffect(() => {
    test_dimensions();
  }, []);

  // every time the window is resized, the timer is cleared and set again
  // the net effect is the component will only reset after the window size
  // is at rest for the duration set in RESET_TIMEOUT.  This prevents rapid
  // redrawing of the component for more complex components such as charts
  window.addEventListener('resize', ()=>{
    clearInterval(movement_timer);
    movement_timer = setTimeout(test_dimensions, RESET_TIMEOUT);
  });


  const [image] = useImage(url);

  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    if (!image) {
      return;
    }
    const aspectRatio = image.width / image.height;
    //const scale = Math.min(width / image.width, height / image.height);
    const width = containerWidth;
    const height = width / aspectRatio;
    const newScale = width / image.width;

    setWidth(width);
    setHeight(height);
    setScale(newScale);
    console.log("new width/height", width, height)
    console.log("image", image);
    console.log("stage", <Stage width={100} height={100}></Stage>);
  }, [image, containerWidth]);


  const disp = (
    <div ref={targetRef}>
      <Stage width={width} height={height}>
        <Layer>
          <Image image={image} width={width} height={height} />
        </Layer>
        <Layer>
          {boxes.map(box => {
            console.log("box!", box)
            const x1 = Math.round(box.x1 * scale);
            const y1 = Math.round(box.y1 * scale);
            const x2 = Math.round(box.x2 * scale);
            const y2 = Math.round(box.y2 * scale);

            const label = new Konva.Text({
              text: box.category, fontsize: 12
            })
            const {width: textWidth, height: textHeight} = label.measureSize()
            return (<>
              <Line
                points={[x1, y1, x2, y1, x2, y2, x1, y2]}
                stroke='red'
                strokeWidth={1}
                closed
              />
              <Text
                x={x2-textWidth*2}
                y={y2-textHeight}
                text={box.category}
                fontsize={12}
                fill='red'
              />
            </>
            );
          })}
        </Layer>
      </Stage>
    </div>
  );
  console.log('disp', disp);
  const disp2 = (<span><div><p>hi</p></div></span>);
  console.log('disp2', disp2);
  return disp;
}

export default AnnotatedImage;
