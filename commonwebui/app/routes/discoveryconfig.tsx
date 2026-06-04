import { Form } from 'radix-ui';
import type { Route } from "./+types/discovery";
import React, { useState } from 'react';
import { useNavigate } from "react-router-dom";

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
}: Route.ComponentProps) {

	  const tabIndex = undefined;
    const navigate = useNavigate(); // Hook for navigation
  	  const [portalContainer, setPortalContainer] = React.useState<HTMLDivElement | null>(null);

    // state hook for Discovery object
    const [discoveryState, setDiscoveryState] = useState(
      loaderData
    )

    const updateStripWhiteSpace = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, stripWhiteSpace: !discoveryState.stripWhiteSpace }
      });
    }

    const updateConvertToLower = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, convertToLower: !discoveryState.convertToLower }
      });
    }

    const updateConvertToASCII = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, convertToASCII: !discoveryState.convertToASCII }
      });
    }

    const updateSingleSpaces = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, singleSpaces: !discoveryState.singleSpaces }
      });
    }

    const updateSearchSemanticOriginal = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchSemanticOriginal: !discoveryState.searchSemanticOriginal }
      });
    }

    const updateSearchBM25sOriginal = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchBM25sOriginal: !discoveryState.searchBM25sOriginal }
      });
    }

    const updateSearchSemanticMulti = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchSemanticMulti: !discoveryState.searchSemanticMulti }
      });
    }

    const updateSearchBM25sMulti = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchBM25sMulti: !discoveryState.searchBM25sMulti }
      });
    }

    const updateSearchSemanticRewrite = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchSemanticRewrite: !discoveryState.searchSemanticRewrite }
      });
    }

    const updateSearchBM25sRewrite = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchBM25sRewrite: !discoveryState.searchBM25sRewrite }
      });
    }

    const updateSearchSemanticHyDE = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchSemanticHyDE: !discoveryState.searchSemanticHyDE }
      });
    }

    const updateSearchBM25sHyDE = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, searchBM25sHyDE: !discoveryState.searchBM25sHyDE }
      });
    }

    const updateOutputNumber = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, outputNumber: e.target.value }
      });
    }

    const updateLoadDocument = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, loadDocument: !discoveryState.loadDocument }
      });
    }

    const updateParseChunks = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, parseChunks: !discoveryState.parseChunks }
      });
    }

    const updateMakeRawVector = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, makeRawVector: !discoveryState.makeRawVector }
      });
    }

    const updateBm25Process = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, bm25Process: !discoveryState.bm25Process }
      });
    }

    const updateSearch = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, search: !discoveryState.search }
      });
    }

    const updateClear = () => {
      setDiscoveryState(previousState => {
        return { ...previousState, clear: !discoveryState.clear }
      });
    }

    const updateChunkSize = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, chunkSize: e.target.value }
      });
    }

    const updateChunkOverlap = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, chunkOverlap: e.target.value }
      });
    }

    const updateSemanticRetrieveNumber = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, semanticRetrieveNumber: e.target.value }
      });
    }

    const updateSemanticMaxCutItemDistance = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, semanticMaxCutItemDistance: e.target.value }
      });
    }

    const updateBm25sRetrieveNumber = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, bm25sRetrieveNumber: e.target.value }
      });
    }

    const updateBm25sMinCutOffScore = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, bm25sMinCutOffScore: e.target.value }
      });
    }

    const updateRrfCutOffValue = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, rrfCutOffValue: e.target.value }
      });
    }

    const updateDocumentFolder = (e) => {
      discoveryState.source = []
      setDiscoveryState(previousState => {
        return { ...previousState, documentFolder: e.target.value }
      });

    }

    const updateRagDatapath = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, ragDatapath: e.target.value }
      });
    }

    const updateDataFolder = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, dataFolder: e.target.value }
      });
    }

    const updateBm25IndexFolder = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, bm25IndexFolder: e.target.value }
      });
    }

    const updateOutputFileName = (e) => {
      setDiscoveryState(previousState => {
        return { ...previousState, outputFileName: e.target.value }
      });
    }

    const handleSubmit = async (event) => {
      event.preventDefault();
      const formData = new FormData(event.target);
      try {
        const response = await fetch("http://127.0.0.1:8000/discovery/config", {
          method: "POST",
          body: JSON.stringify(discoveryState),

        });
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        event.target.reset();
      } catch (error) {
        console.error(error);
      }
      navigate("/discovery");
    };

    const handleCancel = async (event) => {
      event.preventDefault();
      navigate("/discovery");
    };
    


//      "rrfOutlierZScoreThreshold": 3,
//      "rrfOutlierIQRCoefficient": 1.5,


    return (
      <main>
				<Theme accentColor="plum" grayColor="sand" appearance="dark" radius="full" scaling="100%" panelBackground="translucent">

        <Flex direction="column">
          <Flex gap="3" mb="2">
            <Heading as="h4" size="3" mb="2" color="plum">Discovery Settings</Heading>
          </Flex>

          <Flex flexShrink="0" gap="4" direction="row">

              <Card size="4" id ="TextProcessingCard">
                <Heading as="h4" size="3" trim="start" mb="2" color="plum">Text Processing</Heading>
                <Box><Separator size="4" my="5" /></Box>
                  <Flex direction="column" gap="3" mt="1">

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="stripWhiteSpace" checked={discoveryState.stripWhiteSpace} onCheckedChange={updateStripWhiteSpace} />
                        <Text>Strip Whitespace</Text>
                      </Text>
                    </Flex>

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="convertToLower" checked = { discoveryState.convertToLower } onCheckedChange={updateConvertToLower} />
                        <Text>Convert to Lower</Text>
                      </Text>
                    </Flex>

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="convertToASCII" checked = { discoveryState.convertToASCII } onCheckedChange={updateConvertToASCII} />
                        <Text>Convert to ASCII</Text>
                      </Text>
                    </Flex>

                    <Flex asChild gap="2">
                      <Text as="label" size="2" weight="bold">
                        <Switch tabIndex={tabIndex} name="singleSpaces" checked = { discoveryState.singleSpaces } onCheckedChange={updateSingleSpaces} />
                        <Text>Single Spaces</Text>
                      </Text>
                    </Flex>
                </Flex>
              </Card>

              <Card size="4" id ="SearchConfigurationCard">
                <Heading as="h4" size="3" trim="start" mb="2" color="plum">Search Configuration</Heading>
                <Box><Separator size="4" my="5" /></Box>

                <Flex direction="column" gap="3" mt="1">
                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticOriginal" checked = { discoveryState.searchSemanticOriginal } onCheckedChange={updateSearchSemanticOriginal} />
                      <Text>Semantic</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sOriginal" checked = { discoveryState.searchBM25sOriginal } onCheckedChange={updateSearchBM25sOriginal} />
                      <Text>Keyword</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticMulti" checked = { discoveryState.searchSemanticMulti } onCheckedChange={updateSearchSemanticMulti} />
                      <Text>Semantic Multi</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sMulti" checked = { discoveryState.searchBM25sMulti } onCheckedChange={updateSearchBM25sMulti} />
                      <Text>Keyword Multi</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticRewrite" checked = { discoveryState.searchSemanticRewrite } onCheckedChange={updateSearchSemanticRewrite} />
                      <Text>Semantic Rewrite</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sRewrite" checked = { discoveryState.searchBM25sRewrite } onCheckedChange={updateSearchBM25sRewrite} />
                      <Text>Keyword Rewrite</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchSemanticHyDE" checked = { discoveryState.searchSemanticHyDE } onCheckedChange={updateSearchSemanticHyDE} />
                      <Text>Semantic HyDE</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="searchBM25sHyDE" checked = { discoveryState.searchBM25sHyDE } onCheckedChange={updateSearchBM25sHyDE} />
                      <Text>Keyword HyDE</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>Output</Text>
                      <Tooltip content="Count of search results stored for output. Count ranges from 1 to 2048. Default is 50.">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.outputNumber} placeholder="1-2048" step="1" onChange={updateOutputNumber} />
                      </Tooltip>
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
                      <Switch tabIndex={tabIndex} name="workflowLoadDocuments" checked = { discoveryState.loadDocument } onCheckedChange={updateLoadDocument} />
                      <Text>Load Documents</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowParseChunks" checked = { discoveryState.parseChunks } onCheckedChange={updateParseChunks} />
                      <Text>Parse Chunks</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowMakeRawVector" checked = { discoveryState.makeRawVector } onCheckedChange={updateMakeRawVector} />
                      <Text>Make Vectors</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowBm25Process" checked = { discoveryState.bm25Process } onCheckedChange={updateBm25Process} />
                      <Text>Make Keywords</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowSearch" checked = { discoveryState.search } onCheckedChange={updateSearch} />
                      <Text>Perform Search</Text>
                    </Text>
                  </Flex>

                  <Flex asChild gap="2">
                    <Text as="label" size="2" weight="bold">
                      <Switch tabIndex={tabIndex} name="workflowClear" checked = { discoveryState.clear } onCheckedChange={updateClear} />
                      <Text>Clean Temp Files</Text>
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
                      <Text>Chunk Size</Text>
                      <Tooltip content="Chunk Size represents number of characters in each text chunk. Chunk size value ranges from 128 to 512. Default is 512">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.chunkSize} placeholder="128-512" step="1" onChange={updateChunkSize} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>Chunk Overlap</Text>
                      <Tooltip content="Chunk Overlap represents number of overlapping characters between two chunks. Chunk overlap value ranges from 0 to 64. Default is 48">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.chunkOverlap} placeholder="0-64" step="1" onChange={updateChunkOverlap} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>Semantic Retrieve</Text>
                      <Tooltip content="Semantic retrieve establishes maximum number of semantic search results. Semantic retrieve value ranges from 0 (no search results are included) to 2048. Default is 50">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.semanticRetrieveNumber} placeholder="0-2048" step="1" onChange={updateSemanticRetrieveNumber} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>Semantic Cut Off</Text>
                      <Tooltip content="Semantic cut off establishes maximum semantic distance for search results. Semantic cut off value ranges from 0.0 (no all search results are included) to 1.0 (all search results). Default is 1.0">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.semanticMaxCutItemDistance} placeholder="0.0-1.0" onChange={updateSemanticMaxCutItemDistance} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>BM25s Retrieve</Text>
                      <Tooltip content="BM25s retrieve establishes maximum number of BM25s search results. BM25s retrieve value ranges from 0 (no search results are included) to 2048. Default is 50">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.bm25sRetrieveNumber} placeholder="0-2048" step="1" onChange={updateBm25sRetrieveNumber} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>BM25s Cut Off</Text>
                      <Tooltip content="BM25s cut off establishes minimum BM25s score for search results. BM25s cut off value is a positive float. Default is 0.0">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.bm25sMinCutOffScore} placeholder=" >= 0" onChange={updateBm25sMinCutOffScore} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text>RRF Cut Off</Text>
                      <Tooltip content="Reciprocal Rank Fusion (RRF) cut off establishes minimum RRF score for search results. RRF cut off value ranges from 0.0 (all search results are included) to 1.0 (no search results). Default is 0.0">
                        <TextField.Root style={{ width: "80px" }} type="number" value = {discoveryState.rrfCutOffValue} placeholder="0.0-1.0" onChange={updateRrfCutOffValue} />
                      </Tooltip>
                    </Text>
                  </Flex>

                </Flex>
              </Card>
            </Flex>

            <Flex flexShrink="0" gap="4" direction="row" py="2">

              <Card size="4"  id ="FoldersConfigurationCard">
                <Heading as="h4" size="3" trim="start" mb="2" color="plum">Folders Configuration</Heading>
                <Box><Separator size="4" my="5" /></Box>
  
                <Flex direction="column" gap="2" mt="1">

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text style={{ width: "110px" }} >Documents</Text>
                      <Tooltip content="Folder with source documents.">
                        <TextField.Root style={{ width: "500px" }} type="url" value = {discoveryState.documentFolder} placeholder="../testdata/discoverydocuments" onChange={updateDocumentFolder} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text style={{ width: "110px" }} >RAG Database</Text>
                      <Tooltip content="Folder with RAG database.">
                        <TextField.Root style={{ width: "500px" }} type="url" value = {discoveryState.ragDatapath} placeholder="../testdata/discoverydocuments/chromadb" onChange={updateRagDatapath} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text style={{ width: "110px" }} >Data Folder</Text>
                      <Tooltip content="Folder with temp files.">
                        <TextField.Root style={{ width: "500px" }} type="url" value = {discoveryState.dataFolder} placeholder="../testdata/discoverydocuments/discoverydata" onChange={updateDataFolder} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold">
                      <Text style={{ width: "110px" }} >BM25s Folder</Text>
                      <Tooltip content="Folder with BM25s index.">
                        <TextField.Root style={{ width: "500px" }} type="url" value = {discoveryState.bm25IndexFolder} placeholder="../testdata/discoverydocuments/__combined.bm25" onChange={updateBm25IndexFolder} />
                      </Tooltip>
                    </Text>
                  </Flex>

                  <Flex asChild gap="4" direction="row" align="center" >
                    <Text as="label" size="2" weight="bold" >
                      <Text style={{ width: "110px" }} >Output File</Text>
                      <Tooltip content="File with search results.">
                        <TextField.Root style={{ width: "500px" }} type="url" value = {discoveryState.outputFileName} placeholder="../testdata/discoverydocuments/discoverydata/DISCOVERY.results.json" onChange={updateOutputFileName} />
                      </Tooltip>
                    </Text>
                  </Flex>

                </Flex>
              </Card>

              <Card size="4">
                <Form.Root className="FormRoot" onSubmit={handleSubmit} >
                    <Button tabIndex={tabIndex} size="2">
                      <Form.Submit>Submit</Form.Submit>
                    </Button>
                    <Button tabIndex={tabIndex} size="2" onClick={handleCancel} >
                      Cancel
                    </Button>
                </Form.Root>
              </Card>

            </Flex>

        </Flex>
			  </Theme>
      </main>
  );
}


