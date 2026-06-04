import { Form } from 'radix-ui';
import type { Route } from "./+types/indexer";
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
  const response = await fetch(`http://127.0.0.1:8000/indexer`);

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

export default function Indexer({
    loaderData,
}: Route.ComponentProps) {

    const tabIndex = undefined;
  
    const [portalContainer, setPortalContainer] = React.useState<HTMLDivElement | null>(null);

    // state hook for Indexer object
    const [indexerState, setindexerState] = useState(
      loaderData
    )

    const updateStripWhiteSpace = () => {
      setindexerState(previousState => {
        return { ...previousState, stripWhiteSpace: !indexerState.stripWhiteSpace }
      });
    }

    const updateConvertToLower = () => {
      setindexerState(previousState => {
        return { ...previousState, convertToLower: !indexerState.convertToLower }
      });
    }

    const updateConvertToASCII = () => {
      setindexerState(previousState => {
        return { ...previousState, convertToASCII: !indexerState.convertToASCII }
      });
    }

    const updateSingleSpaces = () => {
      setindexerState(previousState => {
        return { ...previousState, singleSpaces: !indexerState.singleSpaces }
      });
    }

    const updateLoadDocument = () => {
      setindexerState(previousState => {
        return { ...previousState, loadDocument: !indexerState.loadDocument }
      });
    }

    const updateRawTextFromDocument = () => {
      setindexerState(previousState => {
        return { ...previousState, rawTextFromDocument: !indexerState.rawTextFromDocument }
      });
    }

    const updateFinalJSONfromRaw = () => {
      setindexerState(previousState => {
        return { ...previousState, finalJSONfromRaw: !indexerState.finalJSONfromRaw }
      });
    }

    const updatePrepareBM25corpus = () => {
      setindexerState(previousState => {
        return { ...previousState, prepareBM25corpus: !indexerState.prepareBM25corpus }
      });
    }

    const updateVectorizeFinalJSON = () => {
      setindexerState(previousState => {
        return { ...previousState, vectorizeFinalJSON: !indexerState.vectorizeFinalJSON }
      });
    }

    const updateClear = () => {
      setindexerState(previousState => {
        return { ...previousState, clear: !indexerState.clear }
      });
    }

    const handleSubmit = async (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);
      try {
        const response = await fetch("http://127.0.0.1:8000/indexer/config", {
          method: "POST",
          body: JSON.stringify(indexerState),

        });
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        event.target.reset();
      } catch (error) {
        console.error(error);
      }
    };
  
    return (
      <main>
        <Theme accentColor="plum" grayColor="sand" appearance="dark" radius="full" scaling="100%" panelBackground="translucent">
          <Flex flexShrink="0" gap="4" direction="row">

              <Card size="4" id ="TextProcessingCard">
                <Heading as="h4" size="3" trim="start" mb="2" color="plum">Text Processing</Heading>
                <Box><Separator size="4" my="5" /></Box>
                  <Flex direction="column" gap="3" mt="1">

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="stripWhiteSpace" checked={indexerState.stripWhiteSpace} onCheckedChange={updateStripWhiteSpace} />
                        <Text>Strip Whitespace</Text>
                      </Text>
                    </Flex>

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="convertToLower" checked = { indexerState.convertToLower } onCheckedChange={updateConvertToLower} />
                        <Text>Convert to Lower</Text>
                      </Text>
                    </Flex>

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="convertToASCII" checked = { indexerState.convertToASCII } onCheckedChange={updateConvertToASCII} />
                        <Text>Convert to ASCII</Text>
                      </Text>
                    </Flex>

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="singleSpaces" checked = { indexerState.singleSpaces } onCheckedChange={updateSingleSpaces} />
                        <Text>Single Spaces</Text>
                      </Text>
                    </Flex>
                </Flex>
              </Card>

              <Card size="4" id ="PhasesCard">
                <Heading as="h4" size="3" trim="start" mb="2" color="plum">Workflow Phases</Heading>
                <Box><Separator size="4" my="5" /></Box>

                <Flex direction="column" gap="3" mt="1">

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowLoadDocuments" checked = { indexerState.loadDocument } onCheckedChange={updateLoadDocument} />
                      <Text>Load Documents</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowRawTextFromDocument" checked = { indexerState.rawTextFromDocument } onCheckedChange={updateRawTextFromDocument} />
                      <Text>Raw JSON</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowFinalJSONfromRaw" checked = { indexerState.finalJSONfromRaw } onCheckedChange={updateFinalJSONfromRaw} />
                      <Text>Final JSON</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowPrepareBM25corpus" checked = { indexerState.prepareBM25corpus } onCheckedChange={updatePrepareBM25corpus} />
                      <Text>Make Keywords</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowVectorizeFinalJSON" checked = { indexerState.vectorizeFinalJSON } onCheckedChange={updateVectorizeFinalJSON} />
                      <Text>Make Vectors</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowClear" checked = { indexerState.clear } onCheckedChange={updateClear} />
                      <Text>Clean Temp Files</Text>
                    </Text>
                  </Flex>

                </Flex>
              </Card>

          </Flex>

          <Form.Root className="FormRoot" onSubmit={handleSubmit} >
            <Form.Field name="data" hidden >
              <Form.Control value={JSON.stringify(indexerState)} />
            </Form.Field>
            <Form.Submit>Submit</Form.Submit>
          </Form.Root>

        </Theme>
      </main>
  );
}


