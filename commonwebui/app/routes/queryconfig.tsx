import { Form } from 'radix-ui';
import type { Route } from "./+types/query";
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
  const response = await fetch(`http://127.0.0.1:8000/query`);

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

export default function Query({
    loaderData,
}: Route.ComponentProps) {

    const tabIndex = undefined;
  
    const [portalContainer, setPortalContainer] = React.useState<HTMLDivElement | null>(null);

    // state hook for Indexer object
    const [queryState, setQueryState] = useState(
      loaderData
    )

    const updateSearchSemanticOriginal = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticOriginal: !queryState.searchSemanticOriginal }
      });
    }

    const updateSearchSemanticOriginalCompress = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticOriginalCompress: !queryState.searchSemanticOriginalCompress }
      });
    }

    const updateSearchSemanticHyDE = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticHyDE: !queryState.searchSemanticHyDE }
      });
    }

    const updateSearchSemanticHyDECompress = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticHyDECompress: !queryState.searchSemanticHyDECompress }
      });
    }

    const updateSearchSemanticMulti = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticMulti: !queryState.searchSemanticMulti }
      });
    }

    const updateSearchSemanticMultiCompress = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticMultiCompress: !queryState.searchSemanticMultiCompress }
      });
    }

    const updateSearchSemanticRewrite = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticRewrite: !queryState.searchSemanticRewrite }
      });
    }

    const updateSearchSemanticRewriteCompress = () => {
      setQueryState(previousState => {
        return { ...previousState, searchSemanticRewriteCompress: !queryState.searchSemanticRewriteCompress }
      });
    }    

    const updateSearchBM25sOriginal = () => {
      setQueryState(previousState => {
        return { ...previousState, searchBM25sOriginal: !queryState.searchBM25sOriginal }
      });
    }

    const updateSearchBM25sOriginalCompress = () => {
      setQueryState(previousState => {
        return { ...previousState, searchBM25sOriginalCompress: !queryState.searchBM25sOriginalCompress }
      });
    }    

    const updateSearchBM25sPrep = () => {
      setQueryState(previousState => {
        return { ...previousState, searchBM25sPrep: !queryState.searchBM25sPrep }
      });
    }

    const updateSearchBM25sPrepCompress = () => {
      setQueryState(previousState => {
        return { ...previousState, searchBM25sPrepCompress: !queryState.searchBM25sPrepCompress }
      });
    }    

    const updateOutputNumber = (e) => {
      setQueryState(previousState => {
        return { ...previousState, outputNumber: e.target.value }
      });
    }

    const updateSemanticRetrieveNumber = (e) => {
      setQueryState(previousState => {
        return { ...previousState, semanticRetrieveNumber: e.target.value }
      });
    }

    const updateSemanticMaxCutItemDistance = (e) => {
      setQueryState(previousState => {
        return { ...previousState, semanticMaxCutItemDistance: e.target.value }
      });
    }

    const updateBm25sRetrieveNumber = (e) => {
      setQueryState(previousState => {
        return { ...previousState, bm25sRetrieveNumber: e.target.value }
      });
    }

    const updateBm25sMinCutOffScore = (e) => {
      setQueryState(previousState => {
        return { ...previousState, bm25sMinCutOffScore: e.target.value }
      });
    }

    const updateTokenizerStopWordsEn = () => {
      setQueryState(previousState => {
        return { ...previousState, tokenizerStopWordsEn: !queryState.tokenizerStopWordsEn }
      });
    }

    const updateTokenizerStemmer = () => {
      setQueryState(previousState => {
        return { ...previousState, tokenizerStemmer: !queryState.tokenizerStemmer }
      });
    }

    const updateQueryPreprocess = () => {
      setQueryState(previousState => {
        return { ...previousState, queryPreprocess: !queryState.queryPreprocess }
      });
    }

    const handleSubmit = async (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);
      try {
        const response = await fetch("http://127.0.0.1:8000/query/config", {
          method: "POST",
          body: JSON.stringify(queryState),

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

              <Card size="4" id ="SearchConfigurationCard">
                <Heading as="h4" size="3" trim="start" mb="2" color="plum">Search Configuration</Heading>
                <Box><Separator size="4" my="5" /></Box>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticOriginal" checked = { queryState.searchSemanticOriginal } onCheckedChange={updateSearchSemanticOriginal} />
                      <Text>Semantic</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticOriginalCompress" checked = { queryState.searchSemanticOriginalCompress } onCheckedChange={updateSearchSemanticOriginalCompress} />
                      <Text>Semantic Compress</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticHyDE" checked = { queryState.searchSemanticHyDE } onCheckedChange={updateSearchSemanticHyDE} />
                      <Text>Semantic HyDE</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticHyDECompress" checked = { queryState.searchSemanticHyDECompress } onCheckedChange={updateSearchSemanticHyDECompress} />
                      <Text>Semantic HyDE Compress</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticMulti" checked = { queryState.searchSemanticMulti } onCheckedChange={updateSearchSemanticMulti} />
                      <Text>Semantic Multi</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticMultiCompress" checked = { queryState.searchSemanticMultiCompress } onCheckedChange={updateSearchSemanticMultiCompress} />
                      <Text>Semantic Multi Compress</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticRewrite" checked = { queryState.searchSemanticRewrite } onCheckedChange={updateSearchSemanticRewrite} />
                      <Text>Semantic Rewrite</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticRewriteCompress" checked = { queryState.searchSemanticRewriteCompress } onCheckedChange={updateSearchSemanticRewriteCompress} />
                      <Text>Semantic Rewrite Compress</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sOriginal" checked = { queryState.searchBM25sOriginal } onCheckedChange={updateSearchBM25sOriginal} />
                      <Text>Keyword</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sOriginalCompress" checked = { queryState.searchBM25sOriginalCompress } onCheckedChange={updateSearchBM25sOriginalCompress} />
                      <Text>Keyword Compress</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sPrep" checked = { queryState.searchBM25sPrep } onCheckedChange={updateSearchBM25sPrep} />
                      <Text>Keyword Prep</Text>
                    </Text>
                  </Flex>
                </Flex>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sPrepCompress" checked = { queryState.searchBM25sPrepCompress } onCheckedChange={updateSearchBM25sPrepCompress} />
                      <Text>Keyword Prep Compress</Text>
                    </Text>
                  </Flex>
                </Flex>

              </Card>

              <Card size="4" id ="AdvancedConfigurationCard">
                <Heading as="h4" size="3" trim="start" mb="2" color="plum">Advanced Configuration</Heading>
                <Box><Separator size="4" my="5" /></Box>

                <Flex direction="column" gap="2" mt="1">

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>Semantic Retrieve</Text>
                      <Tooltip content="Semantic retrieve establishes maximum number of semantic search results. Semantic retrieve value ranges from 0 (no search results are included) to 2048. Default is 50">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {queryState.semanticRetrieveNumber} placeholder="0-2048" step="1" onChange={updateSemanticRetrieveNumber} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>Semantic Cut Off</Text>
                      <Tooltip content="Semantic cut off establishes maximum semantic distance for search results. Semantic cut off value ranges from 0.0 (no all search results are included) to 1.0 (all search results). Default is 1.0">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {queryState.semanticMaxCutItemDistance} placeholder="0.0-1.0" onChange={updateSemanticMaxCutItemDistance} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>BM25s Retrieve</Text>
                      <Tooltip content="BM25s retrieve establishes maximum number of BM25s search results. BM25s retrieve value ranges from 0 (no search results are included) to 2048. Default is 50">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {queryState.bm25sRetrieveNumber} placeholder="0-2048" step="1" onChange={updateBm25sRetrieveNumber} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>BM25s Cut Off</Text>
                      <Tooltip content="BM25s cut off establishes minimum BM25s score for search results. BM25s cut off value is a positive float. Default is 0.0">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {queryState.bm25sMinCutOffScore} placeholder=" >= 0" onChange={updateBm25sMinCutOffScore} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>Output</Text>
                      <Tooltip content="Count of search results stored for output. Count ranges from 1 to 2048. Default is 50.">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {queryState.outputNumber} placeholder="1-2048" step="1" onChange={updateOutputNumber} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex direction="column" gap="3" mt="1">
                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="switchTokenizerStopWordsEn" checked = { queryState.tokenizerStopWordsEn } onCheckedChange={updateTokenizerStopWordsEn} />
                        <Text>Tokenizer Stop Words</Text>
                      </Text>
                    </Flex>
                  </Flex>

                  <Flex direction="column" gap="3" mt="1">
                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="switchTokenizerStemmer" checked = { queryState.tokenizerStemmer } onCheckedChange={updateTokenizerStemmer} />
                        <Text>Tokenizer Stemmer</Text>
                      </Text>
                    </Flex>
                  </Flex>

                  <Flex direction="column" gap="3" mt="1">
                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="switchPreprocessQuery" checked = { queryState.queryPreprocess } onCheckedChange={updateQueryPreprocess} />
                        <Text>Preprocess Query</Text>
                      </Text>
                    </Flex>
                  </Flex>


                </Flex>
              </Card>


          </Flex>

          <Form.Root className="FormRoot" onSubmit={handleSubmit} >
            <Form.Field name="data" hidden >
              <Form.Control value={JSON.stringify(queryState)} />
            </Form.Field>
            <Form.Submit>Submit</Form.Submit>
          </Form.Root>

        </Theme>
      </main>
  );
}
