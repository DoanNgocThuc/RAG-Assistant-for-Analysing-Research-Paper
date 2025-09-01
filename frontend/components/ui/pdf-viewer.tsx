import { Document, Page, pdfjs } from "react-pdf";
import { useState, Dispatch, SetStateAction, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PDFViewerProps {
  pdfFile: File | string | null;
  pageNumber: number;
  setPageNumber: Dispatch<SetStateAction<number>>;
  numPages: number | null;
  setNumPages: Dispatch<SetStateAction<number | null>>;
  isLoadingPdf: boolean;
  highlightText: string;
}

export default function PDFViewer({
  pdfFile,
  pageNumber,
  setPageNumber,
  numPages,
  setNumPages,
  isLoadingPdf,
  highlightText,
}: PDFViewerProps & { highlightText?: string }) {
  const pageRef = useRef<HTMLDivElement>(null);
  const [isPageRendered, setIsPageRendered] = useState(false);
  const [containerWidth, setContainerWidth] = useState<number | undefined>(
    undefined
  );

  useEffect(() => {
    function updateWidth() {
      if (pageRef.current) {
        setContainerWidth(pageRef.current.offsetWidth);
      }
    }
    updateWidth();
    window.addEventListener("resize", updateWidth);
    return () => window.removeEventListener("resize", updateWidth);
  }, []);

  useEffect(() => {
    setTimeout(() => {
      if (!highlightText || !isPageRendered) return;
      const textLayer = pageRef.current?.querySelector(
        ".react-pdf__Page__textContent"
      );
      if (textLayer) {
        try {
          const spans = Array.from(
            textLayer.querySelectorAll("span[role='presentation']")
          );
          if (spans.length > 0 && /^\d+$/.test(spans[0].innerHTML.trim())) {
            spans.shift();
          }
          if (
            spans.length > 0 &&
            /^\d+$/.test(spans[spans.length - 1].innerHTML.trim())
          ) {
            spans.pop();
          }
          const normalize = (str: string) =>
            str.replace(/[\s\n\r]+/g, "").replace(/ /g, "");
          const fullTextNorm = normalize(
            spans.map((s) => s.innerHTML).join("")
          );
          // console.log("Full text normalized:", fullTextNorm);
          let highlightNorm = normalize(highlightText).replace(/\d+$/, "");
          // console.log("Highlight text normalized:", highlightNorm);

          // Tìm vị trí khớp toàn bộ
          let idx = fullTextNorm.indexOf(highlightNorm);

          // Nếu không khớp toàn bộ, thử khớp từng phần nhỏ (chia theo dấu chấm, xuống dòng, hoặc dấu phẩy)
          if (idx === -1) {
            const parts = highlightNorm
              .split(/[.,;:]/)
              .map((p) => p.trim())
              .filter(Boolean);
            for (const part of parts) {
              if (part.length < 10) continue; // bỏ qua cụm quá ngắn
              const partIdx = fullTextNorm.indexOf(part);
              if (partIdx !== -1) {
                idx = partIdx;
                highlightNorm = part;
                break;
              }
            }
          }

          if (idx === -1) {
            console.warn("Highlight text not found in page text.");
            return;
          }

          let currIdx = 0;
          spans.forEach((span) => {
            const spanText = normalize(span.innerHTML);
            const spanStart = currIdx;
            const spanEnd = currIdx + spanText.length;
            if (spanEnd > idx && spanStart < idx + highlightNorm.length) {
              const highlightStart = Math.max(0, idx - spanStart);
              const highlightEnd = Math.min(
                spanText.length,
                idx + highlightNorm.length - spanStart
              );
              const before = span.innerHTML.slice(0, highlightStart);
              const highlight = span.innerHTML.slice(
                highlightStart,
                highlightEnd
              );
              const after = span.innerHTML.slice(highlightEnd);
              span.innerHTML = `${before}<mark style="background:yellow">${highlight}</mark>${after}`;
            }
            currIdx += spanText.length;
          });
        } catch (e) {
          console.warn("Highlight error:", e);
        }
      }
    }, 2000);
  }, [highlightText, isPageRendered]);

  if (!pdfFile) return null;
  return (
    <div
      ref={pageRef}
      className="w-full h-full flex flex-col items-center justify-center"
    >
      {pdfFile && (
        <Document
          file={pdfFile}
          onLoadSuccess={({ numPages }) => setNumPages(numPages)}
          loading={<div>Loading PDF...</div>}
          error={<div>Failed to load PDF</div>}
        >
          <Page
            pageNumber={pageNumber}
            onRenderSuccess={() => setIsPageRendered(true)}
            width={containerWidth}
          />
        </Document>
      )}

      <div className="flex gap-2 items-center">
        <Button
          size="sm"
          className="h-6 w-6 px-1 py-1 text-xs"
          disabled={pageNumber <= 1}
          onClick={() => setPageNumber((p: number) => Math.max(1, p - 1))}
        >
          <ChevronLeft />
        </Button>
        <span className="text-sm text-muted-foreground flex items-center">
          Page {pageNumber} / {numPages || "?"}
        </span>
        <Button
          size="sm"
          className="h-6 w-6 px-1 py-1 text-xs"
          disabled={numPages ? pageNumber >= numPages : true}
          onClick={() =>
            setPageNumber((p: number) =>
              numPages ? Math.min(numPages, p + 1) : p
            )
          }
        >
          <ChevronRight />
        </Button>
      </div>
    </div>
  );
}
