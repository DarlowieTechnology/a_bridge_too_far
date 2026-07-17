import React from "react";
import * as ScrollArea from "@radix-ui/react-scroll-area";
import "./columnstyles.css";

let statusString1= "Saabaaaaaaaaaaaaaa444aaa|Volvo|BMW|Ford";
let statusString2= "Saab|Volvo|BMW|Ford";
let statusString3= "Saabaaaaaaaaa444aaa|Volvo|BMW|Ford";
let statusString4= "Saabaa444aaa|Volvo|BMW|Ford";
const statusLog = [statusString1, statusString2, statusString3, statusString4]

const Messages = Array.from(statusLog);

export default function App() {
  return (
    <ScrollArea.Root className="ScrollAreaRoot">
      <ScrollArea.Viewport className="ScrollAreaViewport">
        {Messages.map((oneRow) => (
            <div>
                {(Array.from(oneRow.split("|"))).map((colVal) => (
                    <div className="Column" >
                        {colVal}
                    </div>
                ))}
            </div>
        ))}
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar orientation="horizontal" className="Scrollbar">
        <ScrollArea.Thumb className="Thumb" />
      </ScrollArea.Scrollbar>
      <ScrollArea.Corner className="Corner" />
    </ScrollArea.Root>
  );
}
