import { Form } from 'radix-ui';
import type { Route } from "./+types/discovery";
import React, {  useState } from 'react';
import { useTimeout  } from 'react-timing-hooks'


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
  ScrollArea,
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


	  const tabIndex = undefined;
  
	  const [portalContainer, setPortalContainer] = React.useState<HTMLDivElement | null>(null);

    // state hook for Discovery object
    const [discoveryState, setDiscoveryState] = useState(
      loaderData,
    )

    const MessagesLog = Array.from(discoveryState.statusLog);

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
      } catch (error) {
          console.error(error);
      }
    };

    const handleRefresh = async (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);

      try {
        const response = await fetch("http://127.0.0.1:8000/discovery", {
          method: "GET"
        });
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        const loaderData = await response.json()
        setDiscoveryState(loaderData)

        event.target.reset();
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
                <ScrollArea style={{ height: "600px", background : "grey" }} >
                  <Box p="1" pr="4">
                    {MessagesLog.map((tag) => (
                      <div className="Tag" key={tag}>
                        {tag}
                      </div>
                    ))}
                  </Box>
                </ScrollArea>
              </Tabs.Content>

              <Tabs.Content value="DocumentList">
                <ScrollArea style={{ height: "600px", background : "grey" }} >
                  <Box p="1" pr="4">
                    {MessagesDoc.map((tag) => (
                      <div className="Tag" key={tag}>
                        {tag}
                      </div>
                    ))}
                  </Box>
                </ScrollArea>
              </Tabs.Content>

              <Tabs.Content value="SearchResults">
                {/* Search Results */}
                <ScrollArea style={{ height: "600px", background : "grey" }} >
                  <Box p="1" pr="4">
                    {TestMessagesSearch.map((tag) => (
                      <div className="Tag" key={tag}>
                        {tag}
                      </div>
                    ))}
                  </Box>
                </ScrollArea>
              </Tabs.Content>

            </Box>
          </Tabs.Root>

          <Form.Root className="FormRoot" name="formRefresh" onSubmit={handleRefresh} >
            <Flex gap="3" mb="2">
              <Form.Submit tabIndex={tabIndex} size="1">
                <Button tabIndex={tabIndex} size="1">
                  Refresh
                </Button>
              </Form.Submit>
            </Flex>
          </Form.Root>

			  </Theme>
      </main>
  );
}


