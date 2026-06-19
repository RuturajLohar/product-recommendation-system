import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const [inputPath, outputPath, previewDir = "/tmp/enriched-workbook-previews"] = process.argv.slice(2);
if (!inputPath || !outputPath) {
  throw new Error("Usage: build_enriched_workbook.mjs INPUT.csv OUTPUT.xlsx [PREVIEW_DIR]");
}

const csvText = await fs.readFile(inputPath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Products" });
const products = workbook.worksheets.getItem("Products");

products.freezePanes.freezeRows(1);
products.showGridLines = false;
const productTable = products.tables.add("A1:Q50001", true, "EnrichedProducts");
productTable.style = "TableStyleMedium2";
productTable.showFilterButton = true;

const header = products.getRange("A1:Q1");
header.format.fill = "#17324D";
header.format.font = { bold: true, color: "#FFFFFF", size: 11 };
header.format.rowHeightPx = 30;
header.format.wrapText = true;

products.getRange("A:A").format.columnWidthPx = 105;
products.getRange("B:B").format.columnWidthPx = 320;
products.getRange("C:C").format.columnWidthPx = 125;
products.getRange("D:E").format.columnWidthPx = 180;
products.getRange("F:F").format.columnWidthPx = 380;
products.getRange("G:G").format.columnWidthPx = 310;
products.getRange("H:H").format.columnWidthPx = 280;
products.getRange("I:I").format.columnWidthPx = 310;
products.getRange("J:N").format.columnWidthPx = 90;
products.getRange("O:O").format.columnWidthPx = 110;
products.getRange("P:Q").format.columnWidthPx = 270;
products.getRange("J2:J50001").setNumberFormat("0.00");
products.getRange("K2:K50001").setNumberFormat("0.0");
products.getRange("L2:L50001").setNumberFormat("0");
products.getRange("M2:M50001").setNumberFormat("0.00");

const guide = workbook.worksheets.add("Dataset Guide");
guide.showGridLines = false;
guide.mergeCells("A1:F2");
guide.getRange("A1").values = [["Intelligent Synthetic Product Dataset"]];
guide.getRange("A1:F2").format.fill = "#17324D";
guide.getRange("A1:F2").format.font = { bold: true, color: "#FFFFFF", size: 18 };
guide.getRange("A1:F2").format.horizontalAlignment = "center";
guide.getRange("A1:F2").format.verticalAlignment = "center";

guide.getRange("A4:B4").values = [["Dataset metric", "Value"]];
guide.getRange("A5:B10").values = [
  ["Products", 50000],
  ["Required columns", 17],
  ["Categories represented", 296],
  ["Recommendation domains", 13],
  ["Reusable synthetic brands", 104],
  ["Generation seed", 42],
];
guide.getRange("A4:B4").format.fill = "#2B6F77";
guide.getRange("A4:B4").format.font = { bold: true, color: "#FFFFFF" };
guide.getRange("A4:B10").format.borders = { preset: "all", style: "thin", color: "#D5DEE8" };
guide.getRange("B5:B10").setNumberFormat("0");

guide.getRange("D4:F4").merge();
guide.getRange("D4").values = [["Important Usage Note"]];
guide.getRange("D4:F4").format.fill = "#E7F2F3";
guide.getRange("D4:F4").format.font = { bold: true, color: "#17324D" };
guide.getRange("D5:F9").merge();
guide.getRange("D5").values = [[
  "Brands and missing enrichment fields are deterministic synthetic demo data. " +
  "Original product identifiers, titles, categories, prices and source URLs are preserved. " +
  "Use this dataset for recommendation-engine development and demonstrations, not as verified Amazon ground truth."
]];
guide.getRange("D5:F9").format.fill = "#F5F8FA";
guide.getRange("D5:F9").format.wrapText = true;
guide.getRange("D5:F9").format.verticalAlignment = "top";
guide.getRange("D4:F9").format.borders = { preset: "outside", style: "thin", color: "#9FB4C2" };

guide.getRange("A12:F12").merge();
guide.getRange("A12").values = [["Data Quality Guarantees"]];
guide.getRange("A12:F12").format.fill = "#2B6F77";
guide.getRange("A12:F12").format.font = { bold: true, color: "#FFFFFF" };
guide.getRange("A13:F17").values = [
  ["50,000 unique product IDs", null, null, null, null, null],
  ["Zero blank, null, Unknown or N/A values", null, null, null, null, null],
  ["At least four features and three specifications per product", null, null, null, null, null],
  ["Ratings constrained to 1-5 and popularity constrained to 0-100", null, null, null, null, null],
  ["Byte-for-byte reproducible output for seed 42", null, null, null, null, null],
];
for (let row = 13; row <= 17; row += 1) guide.mergeCells(`A${row}:F${row}`);
guide.getRange("A13:F17").format.fill = "#F8FAFC";
guide.getRange("A13:F17").format.borders = { preset: "all", style: "thin", color: "#E1E7EC" };
guide.getRange("A:A").format.columnWidthPx = 220;
guide.getRange("B:B").format.columnWidthPx = 110;
guide.getRange("C:C").format.columnWidthPx = 24;
guide.getRange("D:F").format.columnWidthPx = 145;
guide.getRange("1:2").format.rowHeightPx = 32;
guide.getRange("5:10").format.rowHeightPx = 24;
guide.getRange("13:17").format.rowHeightPx = 24;

const productCheck = await workbook.inspect({
  kind: "table",
  range: "Products!A1:E4",
  include: "values,formulas",
  tableMaxRows: 4,
  tableMaxCols: 5,
});
const errorCheck = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});

await fs.mkdir(previewDir, { recursive: true });
const guidePreview = await workbook.render({ sheetName: "Dataset Guide", range: "A1:F17", scale: 1.5 });
await fs.writeFile(path.join(previewDir, "dataset-guide.png"), new Uint8Array(await guidePreview.arrayBuffer()));
const productsPreview = await workbook.render({ sheetName: "Products", range: "A1:I12", scale: 1 });
await fs.writeFile(path.join(previewDir, "products.png"), new Uint8Array(await productsPreview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(productCheck.ndjson);
console.log(errorCheck.ndjson);
console.log(JSON.stringify({ outputPath, previewDir }));
