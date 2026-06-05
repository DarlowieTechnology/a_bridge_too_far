import { Form } from 'radix-ui';
import type { Route } from "./+types/discovery";
import React, {  useState, useEffect, useRef } from 'react';


import "@radix-ui/themes/styles.css";

import {
  Avatar,
	Badge,
	Box,
	Button,
	Card,
  DropdownMenu,
	Flex,
	Grid,
	Heading,
	IconButton,
	Inset,
	Link,
  Section,
	Select,
	Separator,
  Spinner,
	Strong,
  Switch,
  Tabs,
	Text,
	TextArea,
	TextField,
  Tooltip,
	Theme,
} from "@radix-ui/themes";

import { ScrollArea } from "radix-ui";

import {
  ArrowRightIcon,
	BookmarkFilledIcon,
	BookmarkIcon,
	CalendarIcon,
	CrumpledPaperIcon,
  DotsHorizontalIcon,
	FontBoldIcon,
	FontItalicIcon,
	ImageIcon,
	InstagramLogoIcon,
  GearIcon,
  MagicWandIcon,
	MagnifyingGlassIcon,
	RulerHorizontalIcon,
	StrikethroughIcon,
	TextAlignCenterIcon,
	TextAlignLeftIcon,
	TextAlignRightIcon,
	VideoIcon,
} from "@radix-ui/react-icons";



export async function clientLoader({
  params,
}: Route.ClientLoaderArgs) {

  const response = await fetch(`http://127.0.0.1:8000/discovery`);
  if (!response.ok) {
    return "Cannot connect";
  }
  const data = await response.json()
  return data
}

// HydrateFallback is rendered while the client loader is running
export function HydrateFallback() {
  return <div>Loading...</div>;
}



export default function Discovery({
    loaderData,
    ...props
}: Route.ComponentProps) {

    // state hook for Discovery object
    const [discoveryState, setDiscoveryState] = useState(
      loaderData,
    )


    // workflow log viewport
    const logViewportRef = useRef<HTMLDivElement | null>(null);
    const [logCursor, setLogCursor] = useState({ x: 0, y: 0 });
    const [logScroll, setLogScroll] = useState({ top: 0, left: 0 });

    // Scroll to bottom smoothly
    const scrollToBottom = (smooth = true) => {
      const el = logViewportRef.current;
      if (el) {
        el.scrollTo({
          top: el.scrollHeight,
          behavior: smooth ? "smooth" : "auto",
        });
      }
    };

    // Initial scroll after mount
    useEffect(() => {
      requestAnimationFrame(() => scrollToBottom(false));
    }, []);    

    const MessagesLog = Array.from(discoveryState.statusLog);

    // Auto-scroll when new messages arrive
    useEffect(() => {
      scrollToBottom(true);
    }, [MessagesLog]);


	  const tabIndex = undefined;


    const [refreshIsRunning, setRefreshIsRunning] = useState(false)

    // update status log and disable 3 sec refresh if workflow is finished
    async function getData() {
      const url = "http://127.0.0.1:8000/discovery";
      try {
          const response = await fetch(url);
          if (!response.ok) {
              throw new Error(`Response status: ${response.status}`);
          }
          const loaderData = await response.json()
          setRefreshIsRunning(loaderData.inWorkflow)
          setDiscoveryState(loaderData)
      } catch (error) {
          console.error(error.message);
      }
    }

    // refresh status log every 3 sec
    useEffect(() => {
        let interval: NodeJS.Timeout;

        if (refreshIsRunning) {
          interval = setInterval(() => {
              getData();
          }, 3000);
        }
        return () => clearInterval(interval);
    }, [refreshIsRunning]);



    const MessagesDoc = Array.from(discoveryState.source);

    const TestMessagesSearch = Array.from({ length: 50 }).map(
      (_, i, a) => `LOG v1.2.0-beta.${a.length - i}`,
    );


    const handleStart = async (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);

      try {
        const response = await fetch("http://127.0.0.1:8000/discovery/run", {
          method: "GET"
        });
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        const loaderData = await response.json()
        setDiscoveryState(loaderData)

        event.target.reset();

        setRefreshIsRunning(true)

      } catch (error) {
          console.error(error);
      }
    };
   

    return (
      <main>
				<Theme accentColor="plum" grayColor="sand" appearance="dark" radius="full" scaling="100%" panelBackground="translucent">

					<Flex gap="3" mb="2">
            <Heading as="h4" size="3" mb="2" color="plum">Discovery App</Heading>
          </Flex>

          <Form.Root className="FormRoot"  name="formStart" onSubmit={handleStart} >
            <Flex gap="3" mb="2">
              <Box flexGrow="1">
                <TextField.Root
                  tabIndex={tabIndex}
                  size="1"
                  name="searchtext"
                  placeholder="Search query"
                />
              </Box>
              <Form.Submit>
                <Button tabIndex={tabIndex} size="1">
                  <Spinner loading={discoveryState.inWorkflow} >
                    Start
                  </Spinner>
                </Button>
              </Form.Submit>
              <Link href="/discovery/config" size="2" ><GearIcon width="25" height="25" /></Link>
            </Flex>
          </Form.Root>

          <Tabs.Root defaultValue="WorkflowLog">

            <Tabs.List>
              <Tabs.Trigger value="WorkflowLog">
                <Heading size="1" mb="2" color="plum">Workflow Log</Heading>
              </Tabs.Trigger>
              <Tabs.Trigger value="DocumentList">
                <Heading size="1" mb="2" color="plum">Documents</Heading>
              </Tabs.Trigger>
              <Tabs.Trigger value="SearchResults">
                <Heading size="1" mb="2" color="plum">Search Results</Heading>
              </Tabs.Trigger>
            </Tabs.List>

            <Box pt="3">
              <Tabs.Content value="WorkflowLog">
                <ScrollArea.Root>
                  <ScrollArea.Viewport
                    ref={logViewportRef}
                    style={{ padding: 4, height: "600px", background : "grey" }}
                  >
                    {MessagesLog.map((tag) => (
                      <div className="Tag" key={tag}>
                        {tag}
                      </div>
                    ))}
                  </ScrollArea.Viewport>
                  <ScrollArea.Scrollbar orientation="vertical">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Scrollbar orientation="horizontal">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Corner />
                </ScrollArea.Root>
              </Tabs.Content>

              <Tabs.Content value="DocumentList">
                <ScrollArea.Root>
                  <ScrollArea.Viewport
                    ref={logViewportRef}
                    style={{ padding: 4, height: "600px", background : "grey" }}
                  >
                      {MessagesDoc.map((tag) => (
                        <div className="Tag" key={tag}>
                          {tag}
                        </div>
                      ))}
                  </ScrollArea.Viewport>
                  <ScrollArea.Scrollbar orientation="vertical">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Scrollbar orientation="horizontal">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Corner />
                </ScrollArea.Root>
              </Tabs.Content>

              <Tabs.Content value="SearchResults">
                <ScrollArea.Root>
                  <ScrollArea.Viewport
                    ref={logViewportRef}
                    style={{ padding: 4, height: "600px", background : "grey" }}
                  >
                    {TestMessagesSearch.map((tag) => (
                      <div className="Tag" key={tag}>
                        {tag}
                      </div>
                    ))}
                  </ScrollArea.Viewport>
                  <ScrollArea.Scrollbar orientation="vertical">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Scrollbar orientation="horizontal">
                    <ScrollArea.Thumb />
                  </ScrollArea.Scrollbar>
                  <ScrollArea.Corner />
                </ScrollArea.Root>
              </Tabs.Content>

            </Box>
          </Tabs.Root>

			  </Theme>
      </main>
  );
}


