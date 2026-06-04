import { Form } from 'radix-ui';
import type { Route } from "./+types/discovery";
import React, { useState } from 'react';

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
	Strong,
  Switch,
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
  GearIcon,
	ImageIcon,
	InstagramLogoIcon,
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

//  const response = await fetch(`http://127.0.0.1:8000/discovery`);
//  if (!response.ok) {
//    return "Cannot connect";
//  }
//  const data = await response.json()
//  return data

  return ""

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
      {
        "GLOBALllm_Provider": "lmstudio",
        "GLOBALllm_Version": "google/gemma-4-e4b",
        "GLOBALllm_URL": "http://localhost:1234/v1",
        "GLOBALllm_Embed": "nomic-embed-text:latest",
        "GLOBALembedding_URL": "http://localhost:11434/api/embeddings",
        "gemini_key": "",
        "logginglevel": 20,
        "statusFileName": "DISCOVERYLOG",
        "ragDatapath": "../testdata/discoverydocuments/chromadb/",
        "documentFolder": "../testdata/discoverydocuments/",
        "dataFolder": "../testdata/discoverydocuments/discoverydata/",
        "bm25IndexFolder": "../testdata/discoverydocuments/__combined.bm25/",
        "source": [],
        "fileExtensions": [
          "*.txt",
          "*.pdf",
          "*.json"
        ],
        "chunkSize": "256",
        "chunkOverlap": "48",
        "stripWhiteSpace": true,
        "convertToLower": true,
        "convertToASCII": true,
        "singleSpaces": true,
        "loadDocument": true,
        "parseChunks": true,
        "makeRawVector": true,
        "bm25Process": true,
        "search": true,
        "clear": false,
        "query": [],
        "searchSemanticOriginal": true,
        "searchBM25sOriginal": true,
        "searchSemanticMulti": true,
        "searchBM25sMulti": true,
        "searchSemanticRewrite": true,
        "searchBM25sRewrite": true,
        "searchSemanticHyDE": true,
        "searchBM25sHyDE": true,
        "semanticRetrieveNumber": "50",
        "semanticMaxCutItemDistance": "1.0",
        "bm25sRetrieveNumber": "50",
        "bm25sMinCutOffScore": "0.0",
        "rrfCutOffValue": "0.0",
        "rrfOutlierZScoreThreshold": 3,
        "rrfOutlierIQRCoefficient": 1.5,
        "outputNumber": "50",
        "outputFileName": ""
      }
   )

    const TAGS = Array.from({ length: 50 }).map(
      (_, i, a) => `v1.2.0-beta.${a.length - i}`,
    );

    const [visible, setVisible] = useState(true);



    return (
      <main>
				<Theme accentColor="plum" grayColor="sand" appearance="light" radius="full" scaling="100%" panelBackground="translucent">

          <div style={{ padding: "20px" }}>
            {/* Toggle Button */}
            <Button onClick={() => setVisible((prev) => !prev)}>
              {visible ? "Hide Card" : "Show Card"}
            </Button>

            {/* Conditional Rendering */}
            {visible && (
              <Card style={{ marginTop: "20px", maxWidth: "300px" }}>
                <Text as="div" size="3" weight="bold">
                  Radix UI Card
                </Text>
                <Text as="p" size="2" color="gray">
                  This is a simple card that can be shown or hidden.
                </Text>
              </Card>
            )}
          </div>





        <Flex direction="column" gap="3" p="4">
          {/* Toggle button */}
          <Button onClick={() => setVisible((v) => !v)}>
            {visible ? "Hide" : "Show"} Flex Box
          </Button>

          {/* Flex container with conditional display */}
          <Flex
            display={visible ? "flex" : "none"} // hide/show
            gap="2"
            p="3"
            style={{ background: "#f0f0f0", borderRadius: "8px" }}
          >
            <Box p="2" style={{ background: "#ddd" }}>
              Item 1
            </Box>
            <Box p="2" style={{ background: "#ccc" }}>
              Item 2
            </Box>
            <Box p="2" style={{ background: "#bbb" }}>
              Item 3
            </Box>
          </Flex>
        </Flex>





          <Flex>FLEX direction="row" display="inline-flex"</Flex>
          <Flex gap="3" direction="row" style={{ backgroundColor: "gray" }}  display="inline-flex">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Flex>FLEX direction="row" display="none"  (HIDDEN!)</Flex>
          <Flex gap="3" direction="row" style={{ backgroundColor: "gray" }}  display="none">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Flex>FLEX direction="row" display="flex"</Flex>
          <Flex gap="3" direction="row" style={{ backgroundColor: "gray" }}  display="flex">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Box><Separator size="4" my="3" /></Box>

          <Flex>FLEX direction="row" display="flex" justify="start"</Flex>
          <Flex gap="3" direction="row" style={{ backgroundColor: "gray" }}  display="flex" justify="start">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Flex>FLEX direction="row" display="flex" justify="center"</Flex>
          <Flex gap="3" direction="row" style={{ backgroundColor: "gray" }}  display="flex" justify="center">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Flex>FLEX direction="row" display="flex" justify="end"</Flex>
          <Flex gap="3" direction="row" style={{ backgroundColor: "gray" }}  display="flex" justify="end">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Flex>FLEX direction="row" display="flex" justify="between"</Flex>
          <Flex gap="3" direction="row" style={{ backgroundColor: "gray" }}  display="flex" justify="between">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Box><Separator size="4" my="3" /></Box>

          <Flex>FLEX gap="9" box 96px-16px direction="row" display="inline-flex" justify="between" wrap="nowrap"</Flex>
          <Flex gap="9" direction="row" style={{ backgroundColor: "gray" }}  display="inline-flex" justify="between" wrap="nowrap">
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Flex>FLEX gap="9" box 96px-16px  direction="row" display="inline-flex" justify="between" wrap="wrap"</Flex>
          <Flex gap="9" direction="row" style={{ backgroundColor: "gray" }}  display="inline-flex" justify="between" wrap="wrap">
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Box><Separator size="4" my="3" /></Box>

          <Flex>FLEX gap="9" box 96px-16px  direction="row" display="inline-flex"</Flex>
          <Flex gap="9" direction="row" style={{ backgroundColor: "gray" }}  py="2" display="inline-flex">
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="96px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Box><Separator size="4" my="3" /></Box>

          <Flex>FLEX gap="1" direction="column" display="inline-flex"</Flex>
          <Flex gap="1" direction="column" style={{ backgroundColor: "gray" }}  display="inline-flex">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

          <Flex>FLEX gap="1" direction="column" display="flex"</Flex>
          <Flex gap="1" direction="column" style={{ backgroundColor: "gray" }}  display="flex">
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
            <Box width="16px" height="16px" style={{ backgroundColor: "indigo" }} >
            </Box>
          </Flex>

					<Flex gap="3" mb="2">
						<Box flexGrow="1">
							<TextField.Root
								tabIndex={tabIndex}
								size="1"
								placeholder="Search query"
							/>
						</Box>
						<Button tabIndex={tabIndex} size="1">
							Search
						</Button>
            <Form.Root className="FormRoot" >
              <Tooltip content="Application Settings">
                <Button tabIndex={tabIndex} size="1">
                  <Form.Submit><GearIcon /></Form.Submit>
                </Button>
              </Tooltip>
            </Form.Root>
					</Flex>
					<Flex gap="2" mb="5" style={{ backgroundColor: "gray" }}>
            <ScrollArea className="ScrollAreaRoot" style={{ height: "200px" }} >
              {TAGS.map((tag) => (
					      <div className="Tag" key={tag}>
						      {tag}
					      </div>
				      ))}
            </ScrollArea>					
          </Flex>
			  </Theme>
      </main>
  );
}


