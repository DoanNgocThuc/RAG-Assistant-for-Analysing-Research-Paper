// components/research/FormulaSheetAndDialog.tsx
"use client";

import React from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useChat } from "./ChatProvider";

export const FormulaSheetAndDialog: React.FC = () => {
  const {
    isFormulaSheetOpen,
    setIsFormulaSheetOpen,
    formulas,
    handleFormulaClick,
    isDialogOpen,
    setIsDialogOpen,
    selectedFormula,
    explanation,
    isExplanationLoading,
  } = useChat();

  return (
    <>
      <Sheet open={isFormulaSheetOpen} onOpenChange={setIsFormulaSheetOpen}>
        <SheetContent className="w-[400px] sm:w-[540px]">
          <SheetHeader>
            <SheetTitle>Formula Insights</SheetTitle>
          </SheetHeader>
          <ScrollArea className="h-[calc(100vh-200px)] mt-6">
            {formulas.length === 0 ? (
              <p className="text-center text-muted-foreground">
                No formulas found or loading...
              </p>
            ) : (
              formulas.map((f, idx) => (
                <div
                  key={idx}
                  className="p-3 border-b cursor-pointer hover:bg-muted transition-colors"
                  onClick={() => handleFormulaClick(f)}
                >
                  <p className="text-sm font-medium">Page {f.page}</p>
                  <p className="font-mono text-sm whitespace-pre-wrap">
                    {f.formula}
                  </p>
                </div>
              ))
            )}
          </ScrollArea>
        </SheetContent>
      </Sheet>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Formula Explanation - Page {selectedFormula?.page}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="font-mono text-lg whitespace-pre-wrap">
              {selectedFormula?.formula}
            </p>
            {isExplanationLoading ? (
              <p className="text-muted-foreground">Generating explanation...</p>
            ) : (
              <p className="text-sm whitespace-pre-wrap">{explanation}</p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};
