# A Bridge Too Far
unstructured to structured data bridge demonstrator

## Introduction

Organizations store structured and unstructured data. Out of these two, structured data is likely to represent organization IP. On the other hand unstructured data is likely to be collected from external data sources.

Data mining unstructured data requires at very least a categorization. Categories come from structured data. In essence we use organization's IP to transform unstructured data to (somewhat) structured data and so increase organization's IP.

The process or categorization requires unstructured data analysis. Historically this process was unproductive due to many reasons. This all changed by Large Language Models (LLMs). We have now an excellent tool for for unstructured data comprehension.

This project aims to show how LLM can be used to bridge the gap between unstructured and structured data.

## Data Analysis Foundation

The unstructured data for this demonstrator is a collection of job ads. Job ads are created in natural language without restrictive template. However, job ads have common traits that are easily recognized by LLM. Following (incomplete) format can be established.

* Role
  * List of products required
  * List of technologies required
  * List of certifications required

All bullet points (position, products, technologies, certifications) can be described as domain knowledge in the form of structured data.

## Data Mining with LLM

We supply job ad as a context to LLM and prompt LLM for matching category. For example we can create a prompt.

```
Given the list of roles {roles}, what is the single role that matches job description better than other roles?
```

Such prompt may result in a role that matches one of roles in our list. In this case we can create a category tag. This process also creates a natural filter, we discard everything that does not interest us.

We can create as many prompts as we want, matching multiple categories down to details. For example.

```
How many year of experience is required with CrowdStrike Falcon on Windows 11 endpoints?
```

## Crossing the Bridge

Once we have sufficient number of category labels extracted from unstructured data we have ability to perform actions based on these labels. Actions could be as simple as generating a report.

```
Analysis of one million recent job ads for IT roles has shown that 63% of all employers expect new hires to train their existing personnel.
```

Or publish forward-looking predictions of the job market dynamics.

```
Our research indicates growing adoption of Azure Logic Apps at the expense of Azure Functions. This trend is likely to persist in the medium timeframe. 
```

Or perform any other action based on the organization IP.


## Technology Stack

* Ollama local LLM host
* Gemma3:1b LLM
* Python 3.12
* SQLite for representation of structured storage



