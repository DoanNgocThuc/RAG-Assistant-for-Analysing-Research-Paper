// components/research/PDFCard.tsx
"use client";

import React from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { FileText, FunctionSquare, Upload } from "lucide-react";
import { useChat } from "./ChatProvider";

const PDFViewer = dynamic(() => import("@/components/ui/pdf-viewer"), { ssr: false });

export const PDFCard: React.FC = () => {
  const {
    pdfFile,
    isLoadingPdf,
    numPages,
    setNumPages,
    pageNumber,
    setPageNumber,
    setIsFormulaSheetOpen,
    activeChat,
  } = useChat();

  return (
    <Card className="flex flex-col max-h-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Research Paper
          </span>
          {activeChat && <Badge variant="outline" className="ml-auto">Active Chat</Badge>}
        </CardTitle>

        {pdfFile && (
          <Button variant="outline" className="mt-2 gap-2" onClick={() => setIsFormulaSheetOpen(true)} disabled={isLoadingPdf}>
            <FunctionSquare className="h-4 w-4" />
            Formula Insights
          </Button>
        )}
      </CardHeader>

      <CardContent className="flex-1 overflow-hidden">
        <ScrollArea className="h-full pr-3">
          {pdfFile ? (
            <div className="space-y-3">
              <div className="p-3 bg-muted rounded-lg">
                <h3 className="font-semibold mb-2">PDF Loaded: {pdfFile.name}</h3>
                <p className="text-sm text-muted-foreground">File size: {(pdfFile.size / 1024 / 1024).toFixed(2)} MB</p>
              </div>

              <div className="aspect-[3/4] bg-muted rounded-lg flex items-center justify-center">
                {isLoadingPdf ? (
                  <div className="text-center">
                    <p className="text-muted-foreground">Loading PDF...</p>
                  </div>
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center">
                    <PDFViewer
                      pdfFile={pdfFile}
                      pageNumber={pageNumber}
                      setPageNumber={setPageNumber as React.Dispatch<React.SetStateAction<number>>}
                      numPages={numPages}
                      setNumPages={setNumPages as React.Dispatch<React.SetStateAction<number | null>>}
                      isLoadingPdf={isLoadingPdf}
                      highlightText={""}
                    />
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <Upload className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-semibold mb-2">No PDF Loaded</h3>
                <p className="text-muted-foreground mb-3">Upload a research paper to get started or select from chat history</p>
                <div className="flex gap-2 justify-center">
                  <Button onClick={() => document.getElementById("pdf-upload")?.click()} className="gap-2">
                    <Upload className="h-4 w-4" />
                    Choose PDF File
                  </Button>
                </div>
              </div>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
};
