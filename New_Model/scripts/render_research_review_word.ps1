param(
    [Parameter(Mandatory = $true)][string]$DocumentPath,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$resolvedDocument = (Resolve-Path -LiteralPath $DocumentPath).Path
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
[void][System.IO.Directory]::CreateDirectory($resolvedOutput)
$pdfPath = Join-Path $resolvedOutput 'review.pdf'
$wordReview = $null
$reviewDocument = $null
try {
    $wordReview = New-Object -ComObject Word.Application
    $wordReview.Visible = $false
    $wordReview.DisplayAlerts = 0
    $wordReview.AutomationSecurity = 3
    $wordReview.Options.PrintBackground = $false
    $reviewDocument = $wordReview.Documents.Open($resolvedDocument, $false, $true)
    $reviewDocument.Repaginate()
    $pageCount = $reviewDocument.ComputeStatistics(2)
    $reviewDocument.ExportAsFixedFormat($pdfPath, 17, $false)
    @{
        renderer = 'Installed Microsoft Word native PDF export'
        reason = 'Packaged render_docx.py was attempted; LibreOffice soffice.exe is unavailable'
        word_version = $wordReview.Version
        document = $resolvedDocument
        pdf = $pdfPath
        pages = $pageCount
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $resolvedOutput 'render_metadata.json') -Encoding UTF8
    Write-Output "Rendered $pageCount pages to $pdfPath"
} finally {
    if ($null -ne $reviewDocument) {
        $reviewDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($reviewDocument)
    }
    if ($null -ne $wordReview) {
        if ($wordReview.Documents.Count -eq 0) { $wordReview.Quit() }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($wordReview)
    }
}
