# report/pdf_generator.py
"""
Sinh PDF báo cáo 1 trang cho 1 rally tennis.
Output: file PDF với header + trajectory map + bảng metrics + insights.
"""
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── Color palette (khớp UI app) ─────────────────────────────────────
COLOR_HEADER_BG   = HexColor("#1a3a6b")
COLOR_ACCENT      = HexColor("#4fc3f7")
COLOR_P1          = HexColor("#2196f3")
COLOR_P2          = HexColor("#f44336")
COLOR_LIGHT_BG    = HexColor("#f0f4fa")
COLOR_TEXT_DARK   = HexColor("#1a3a6b")
COLOR_TEXT_MEDIUM = HexColor("#5a6a90")


def _register_vietnamese_font():
    """Tìm và đăng ký TTF font hỗ trợ tiếng Việt."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("VNFont", path))
                # Đăng ký bold variant nếu có (Arial bold)
                bold_path = path.replace("arial.ttf", "arialbd.ttf")
                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont("VNFont-Bold", bold_path))
                return "VNFont"
            except Exception:
                continue
    return "Helvetica"  # fallback - tiếng Việt sẽ bị broken


class RallyReportPDF:
    """Sinh báo cáo PDF 1 trang cho 1 rally tennis."""

    def __init__(self):
        self.font = _register_vietnamese_font()
        self.styles = self._build_styles()

    def generate(self, rally_stats, insights, trajectory_path,
                 output_path, video_name=None, fps=None):
        """
        Tạo file PDF tại output_path.

        Args:
            rally_stats: dict từ RallyAnalyzer.analyze()
            insights: list str từ generate_insights()
            trajectory_path: đường dẫn ảnh trajectory map (.png)
            output_path: nơi lưu file PDF
            video_name: tên rally (header)
            fps: FPS video (meta info)
        """
        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=1.5*cm, rightMargin=1.5*cm,
            topMargin=1.0*cm, bottomMargin=1.0*cm,
        )

        story = []

        # 1. Header
        story.append(self._build_header(video_name, rally_stats, fps))
        story.append(Spacer(1, 0.35*cm))

        # 2. Trajectory map image
        if trajectory_path and os.path.exists(trajectory_path):
            img = Image(trajectory_path, width=7.5*cm, height=11*cm,
                        kind='proportional')
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 0.3*cm))

        # 3. Metrics table
        story.append(Paragraph("CHỈ SỐ HIỆU SUẤT", self.styles['section']))
        story.append(Spacer(1, 0.15*cm))
        story.append(self._build_metrics_table(rally_stats))
        story.append(Spacer(1, 0.25*cm))

        # 4. Ball end zone highlight
        zone = rally_stats.get('ball_end_zone') or 'N/A'
        story.append(Paragraph(
            f"<b>Vùng bóng kết thúc rally:</b> {zone}",
            self.styles['highlight']
        ))
        story.append(Spacer(1, 0.3*cm))

        # 5. Insights
        story.append(Paragraph("NHẬN XÉT TỰ ĐỘNG", self.styles['section']))
        story.append(Spacer(1, 0.1*cm))

        if insights:
            for ins in insights[:6]:  # cap 6 insight để fit 1 trang
                story.append(Paragraph(f"• {ins}", self.styles['insight']))
                story.append(Spacer(1, 0.08*cm))
        else:
            story.append(Paragraph(
                "<i>Không có nhận xét phù hợp cho rally này.</i>",
                self.styles['body']
            ))

        story.append(Spacer(1, 0.3*cm))

        # 6. Footer
        story.append(self._build_footer())

        # Render PDF
        doc.build(story)
        return output_path

    # ─────────────────────────────────────────────────────────────
    # Style definitions
    # ─────────────────────────────────────────────────────────────

    def _build_styles(self):
        f = self.font
        return {
            'title': ParagraphStyle(
                name='Title', fontName=f, fontSize=15,
                textColor=white, alignment=TA_CENTER, leading=18,
            ),
            'meta': ParagraphStyle(
                name='Meta', fontName=f, fontSize=8.5,
                textColor=COLOR_TEXT_MEDIUM, alignment=TA_CENTER, leading=11,
            ),
            'section': ParagraphStyle(
                name='Section', fontName=f, fontSize=11,
                textColor=COLOR_TEXT_DARK, leading=14,
                borderColor=COLOR_ACCENT, borderWidth=0,
                leftIndent=0, spaceBefore=4,
            ),
            'body': ParagraphStyle(
                name='Body', fontName=f, fontSize=10,
                textColor=black, leading=13,
            ),
            'highlight': ParagraphStyle(
                name='Highlight', fontName=f, fontSize=10.5,
                textColor=COLOR_TEXT_DARK, leading=14,
                backColor=COLOR_LIGHT_BG, borderPadding=8,
                leftIndent=4, rightIndent=4,
            ),
            'insight': ParagraphStyle(
                name='Insight', fontName=f, fontSize=9.5,
                textColor=COLOR_TEXT_DARK, leading=13, leftIndent=12,
            ),
            'footer': ParagraphStyle(
                name='Footer', fontName=f, fontSize=7.5,
                textColor=COLOR_TEXT_MEDIUM, alignment=TA_CENTER, leading=10,
            ),
        }

    # ─────────────────────────────────────────────────────────────
    # Component builders
    # ─────────────────────────────────────────────────────────────

    def _build_header(self, video_name, stats, fps):
        """Header gồm tiêu đề + meta info."""
        title = "TENNIS RALLY ANALYTICS REPORT"

        date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        meta_parts = []
        if video_name:
            meta_parts.append(f"Rally: <b>{video_name}</b>")
        meta_parts.append(f"Ngày: {date_str}")
        meta_parts.append(f"Duration: {stats.get('duration_seconds', 0)}s")
        if fps:
            meta_parts.append(f"FPS: {fps:.1f}")
        meta_parts.append(f"Frames: {stats.get('n_frames', 0)}")
        meta_text = "&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta_parts)

        header_table = Table(
            [[Paragraph(title, self.styles['title'])],
             [Paragraph(meta_text, self.styles['meta'])]],
            colWidths=[18*cm],
            rowHeights=[0.95*cm, 0.55*cm]
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_HEADER_BG),
            ('BACKGROUND', (0, 1), (-1, 1), COLOR_LIGHT_BG),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return header_table

    def _build_metrics_table(self, stats):
        """Bảng so sánh 2 player + dòng cuối movement ratio span."""
        body = self.styles['body']
        cell = lambda txt: Paragraph(txt, body)

        rows = [
            [cell("<b>Metric</b>"),
             cell("<b><font color='#2196f3'>Player 1</font></b>"),
             cell("<b><font color='#f44336'>Player 2</font></b>")],
            [cell("Distance"),
             cell(f"{stats['distances'].get(1, 0):.1f} m"),
             cell(f"{stats['distances'].get(2, 0):.1f} m")],
            [cell("Peak Speed"),
             cell(f"{stats['peak_speeds'].get(1, 0)} km/h"),
             cell(f"{stats['peak_speeds'].get(2, 0)} km/h")],
            [cell("Avg Speed"),
             cell(f"{stats['avg_speeds'].get(1, 0)} km/h"),
             cell(f"{stats['avg_speeds'].get(2, 0)} km/h")],
            [cell("Sprint Count"),
             cell(str(stats['sprint_counts'].get(1, 0))),
             cell(str(stats['sprint_counts'].get(2, 0)))],
            [cell("Court Coverage"),
             cell(f"{stats['coverage_areas'].get(1, 0):.0f} m²"),
             cell(f"{stats['coverage_areas'].get(2, 0):.0f} m²")],
            [cell("Position Style"),
             cell(stats['position_styles'].get(1, '-')),
             cell(stats['position_styles'].get(2, '-'))],
            [cell("<b>Movement Ratio</b>"),
             cell(f"<b>{stats['movement_ratio']:.2f}x</b> "
                  f"(Player {stats['high_runner']} chạy nhiều hơn)"),
             cell("")],
        ]

        table = Table(rows, colWidths=[4.5*cm, 6.5*cm, 6.5*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), COLOR_HEADER_BG),
            ('TEXTCOLOR',  (0, 0), (-1, 0), white),
            ('BACKGROUND', (0, 1), (-1, -1), COLOR_LIGHT_BG),
            ('GRID',       (0, 0), (-1, -1), 0.5, white),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING',  (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING',   (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
            # Span dòng cuối movement ratio
            ('SPAN', (1, -1), (2, -1)),
        ]))
        return table

    def _build_footer(self):
        text = (
            "Tennis Rally Analytics v1.0  |  "
            "Báo cáo sinh tự động  |  "
            "Player tracking · Court detection · Ball tracking · Speed estimation"
        )
        return Paragraph(text, self.styles['footer'])