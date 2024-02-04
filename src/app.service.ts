import { Injectable } from "@nestjs/common";
import * as fs from "fs";

import { OpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { ContextualCompressionRetriever } from "langchain/retrievers/contextual_compression";
import { LLMChainExtractor } from "langchain/retrievers/document_compressors/chain_extract";
import { PromptTemplate } from '@langchain/core/prompts';
import { Ollama } from "@langchain/community/llms/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

@Injectable()
export class AppService {

  async gpt3() {
    const model = new OpenAI({
      modelName: "gpt-3.5-turbo",
    });

    const prompt = new PromptTemplate({
      template: `
      {question}. Responde de forma breve y conciso

      {context}
      `,
      inputVariables:["question", "context"],
    })
    const baseCompressor = LLMChainExtractor.fromLLM(model);
    
    const text = fs.readFileSync("src/dataset/abc-electric-cars.txt", "utf8");
    
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 2500 });
    const docs = await textSplitter.createDocuments([text]);
    
    // Create a vector store from the documents.
    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    
    const retriever = new ContextualCompressionRetriever({
      baseCompressor,
      baseRetriever: vectorStore.asRetriever(),
    });
    
    const retrievedDocs = await retriever.getRelevantDocuments(
      "Cuál es el nombre de la empresa?"
    );
    
    console.log({ retrievedDocs });
    
    return 'Gpt3!';
  }

  async mistral() {
    const ollama = new Ollama({ baseUrl: "http://localhost:11434", model: "mistral" });
    const embeddings = new OllamaEmbeddings({ model: "mistral", baseUrl: "http://localhost:11434" });

    const prompt = new PromptTemplate({
      template: `
      Dada la siguiente pregunta y contexto, extrae cualquier parte del contexto *TAL CUAL* que sea relevante para responder a la pregunta.
      Si ninguna parte del contexto es relevante, devuelve "No cuento con esa información, ¿gusta que lo contacte con un asesor?" y termina la conversación sin agregar texto adicional.
      
      Responde es español y asegúrate de refinar la información.*Nunca* menciones que la información fue extraída de un contexto.

      > Pregunta: {question} 
      > Contexto:
      >>>
      {context}
      >>>
      Respuesta:`,
      inputVariables:["question", "context"],
    })
    const baseCompressor = LLMChainExtractor.fromLLM(ollama, prompt);
    
    const text = fs.readFileSync("src/dataset/abc-electric-cars.txt", "utf8");
    
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 2500 });
    const docs = await textSplitter.createDocuments([text]);
    
    // Create a vector store from the documents.
    const vectorStore = await HNSWLib.fromDocuments(docs, embeddings);
    
    const retriever = new ContextualCompressionRetriever({
      baseCompressor,
      baseRetriever: vectorStore.asRetriever(),
    });
    
    const retrievedDocs = await retriever.getRelevantDocuments(
      "Cuál es el nombre de la empresa?"
    );
    
    console.log({ retrievedDocs });
    return 'mistral!';
  }
}
