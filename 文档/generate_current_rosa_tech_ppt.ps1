param(
    [string]$OutputPath = "D:\codes\ROSA\文档\当前ROSA实现技术汇报_5页.pptx"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Set-BodyText {
    param(
        $Shape,
        [string[]]$Lines,
        [int]$FontSize = 20,
        [string]$FontName = "Microsoft YaHei"
    )

    $bulletLines = foreach ($line in $Lines) {
        "• $line"
    }
    $text = [string]::Join("`r", $bulletLines)
    $Shape.TextFrame.TextRange.Text = $text
    $Shape.TextFrame.TextRange.Font.Name = $FontName
    $Shape.TextFrame.TextRange.Font.Size = $FontSize
    $Shape.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 6
}

function Add-TitleSlide {
    param(
        $Presentation,
        [string]$Title,
        [string]$Subtitle
    )

    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 1)
    $slide.Shapes.Title.TextFrame.TextRange.Text = $Title
    $slide.Shapes.Title.TextFrame.TextRange.Font.Name = "Microsoft YaHei UI"
    $slide.Shapes.Title.TextFrame.TextRange.Font.Size = 28
    $slide.Shapes.Title.TextFrame.TextRange.Font.Bold = -1

    $subtitleShape = $slide.Shapes.Item(2)
    $subtitleShape.TextFrame.TextRange.Text = $Subtitle
    $subtitleShape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $subtitleShape.TextFrame.TextRange.Font.Size = 18
}

function Add-BulletSlide {
    param(
        $Presentation,
        [string]$Title,
        [string[]]$Bullets
    )

    $slide = $Presentation.Slides.Add($Presentation.Slides.Count + 1, 2)
    $slide.Shapes.Title.TextFrame.TextRange.Text = $Title
    $slide.Shapes.Title.TextFrame.TextRange.Font.Name = "Microsoft YaHei UI"
    $slide.Shapes.Title.TextFrame.TextRange.Font.Size = 24
    $slide.Shapes.Title.TextFrame.TextRange.Font.Bold = -1

    $bodyShape = $slide.Shapes.Item(2)
    Set-BodyText -Shape $bodyShape -Lines $Bullets
}

$slides = @(
    @{
        Type = "title"
        Title = "当前 ROSA 实现技术方案"
        Subtitle = "在线地址主线、SAM 状态、注入与运行时工程`r`仓库主入口：train_qwen_llama_vs_rosa_v2.py"
    },
    @{
        Type = "bullets"
        Title = "系统目标与总体架构"
        Bullets = @(
            "目标：在同一套 Qwen/LLaMA 风格 decoder-only 主干上，对比 baseline 与 rosa_fused。",
            "公平性约束：主干结构一致、参数量对齐、初始化一致、训练脚本统一。",
            "当前训练主线：rosa_train_mode = online_seq；旧 doc_local + sam precompute 保留为 reference_precompute。",
            "核心模块：train_qwen_llama_vs_rosa_v2.py、rosa_addressing.py、rosa_runtime.py、rosa_session.py。",
            "实验工具：profile_rosa_online_baseline.py 与 scan_rosa_injection_layers.py。"
        )
    },
    @{
        Type = "bullets"
        Title = "地址生成主线（AddressEngine + Online SAM）"
        Bullets = @(
            "统一入口：RosaAddressEngine.forward_seq() 与 forward_step()，把训练 prefill、推理解码、reference 回归统一到同一地址抽象。",
            "地址模式：reference_backend、online_exact、online_sam；其中 online_sam 使用真正的 suffix automaton state 在线生成地址。",
            "OnlineRosaState 维护在线状态，旧 list-state 以 ExactMatchRosaState 形式保留作 fallback/reference。",
            "doc_local 在线模式默认提供 full doc prefix 左侧 memory，与 reference 路径做逐位置对齐。",
            "工程收益：同一套地址接口可服务训练、评测、profile 与在线 decode。"
        )
    },
    @{
        Type = "bullets"
        Title = "注入路径与运行时结构"
        Bullets = @(
            "模型入口为 RosaFusedLM，主路径是 compute_rosa_address_batch() -> build_rosa_injection_payload() -> forward(..., rosa_payload=...)。",
            "value backend 支持 shared 与 per_layer；per_layer 可为每个注入层维护独立 value store，并从共享 embedding 平滑初始化。",
            "gate 机制由 match_len prior 与可选的 context-aware gate 组成，用当前 hidden state 与 memory value 共同决定注入强度。",
            "层位控制通过 rosa_inject_layer_ids 完成，rosa_layer_sweep.py 可做单层与层对扫描。",
            "运行时已支持 prefetch、staging buffer 与 hot cache，为后续 host memory / mmap 路线做准备。"
        )
    },
    @{
        Type = "bullets"
        Title = "工程验证、性能现状与下一步"
        Bullets = @(
            "一致性回归：online_seq 训练路径与 reference_precompute 已做到 train path address agreement = 1.0、logit diff = 0.0。",
            "测试覆盖：仓库当前已有 56 个测试，覆盖地址引擎、runtime、session、profile 与主训练入口。",
            "当前 timing：baseline step 约 37.9 ms，rosa_fused step 约 124.1 ms，主要慢点集中在地址支路 rosa_addr 约 85.7 ms。",
            "当前收益：session 已统一 prefill + decode 生命周期，prefetch / staging / hot cache 已正式接入主线。",
            "下一步重点：地址支路异步化与 overlap、DocMemory 外部文档记忆、更正式的 value store 与 host memory 路线。"
        )
    }
)

$powerPoint = $null
$presentation = $null

try {
    $outDir = Split-Path -Parent $OutputPath
    if (-not (Test-Path $outDir)) {
        New-Item -ItemType Directory -Path $outDir | Out-Null
    }

    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = -1
    $presentation = $powerPoint.Presentations.Add()

    foreach ($slideData in $slides) {
        if ($slideData.Type -eq "title") {
            Add-TitleSlide -Presentation $presentation -Title $slideData.Title -Subtitle $slideData.Subtitle
        } else {
            Add-BulletSlide -Presentation $presentation -Title $slideData.Title -Bullets $slideData.Bullets
        }
    }

    if (Test-Path $OutputPath) {
        Remove-Item $OutputPath -Force
    }
    $presentation.SaveAs($OutputPath)
}
finally {
    if ($presentation -ne $null) {
        $presentation.Close()
    }
    if ($powerPoint -ne $null) {
        $powerPoint.Quit()
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}

Write-Output "PPT generated: $OutputPath"
