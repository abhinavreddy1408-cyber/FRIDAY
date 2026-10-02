"""
F.R.I.D.A.Y. PDF Generation Utilities (Phase 5C)
================================================
Professional document generation using fpdf2.
Includes modern templates for Resumes and Cover Letters.
Handles duplicate naming and output organization.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from fpdf import FPDF

from friday import config

logger = logging.getLogger(__name__)


@dataclass
class ResumeData:
    name: str
    email: str
    phone: str
    location: str
    summary: str
    experience: list[dict]  # {company, role, dates, bullets: list[str]}
    education: list[dict]  # {institution, degree, dates, gpa: optional}
    skills: list[str]
    projects: list[dict]  # {name, description, tech: list[str]}
    certifications: list[str]


@dataclass
class CoverLetterData:
    candidate_name: str
    candidate_email: str
    candidate_phone: str
    date: str
    hiring_manager: str  # "Hiring Manager" if unknown
    company_name: str
    role_title: str
    paragraphs: list[str]  # 3-4 paragraphs
    closing: str  # "Sincerely" etc.


class PDFChef:
    """Professional PDF layout engine."""

    COLORS = {  # noqa: RUF012
        "header_bg": (25, 25, 112),  # Midnight Blue
        "header_text": (255, 255, 255),  # White
        "accent": (70, 130, 180),  # Steel Blue
        "body_text": (30, 30, 30),  # Near Black
        "light_gray": (220, 220, 220),  # Dividers
        "section_title": (25, 25, 112),  # Midnight Blue
    }

    def __init__(self):
        self.output_dir = Path(config.PDF_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_resume(self, data: ResumeData, filename: str = "Resume.pdf") -> Path:
        """Generate a professional chronological resume."""
        pdf = FPDF()
        pdf.set_title(f"Resume - {data.name}")
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # 1. Header
        self._add_resume_header(pdf, data)
        pdf.ln(5)

        # 2. Summary
        self._add_section_title(pdf, "PROFESSIONAL SUMMARY")
        pdf.set_font("helvetica", size=10)
        pdf.set_text_color(*self.COLORS["body_text"])
        pdf.multi_cell(0, 5, data.summary)
        pdf.ln(5)

        # 3. Work Experience
        self._add_section_title(pdf, "WORK EXPERIENCE")
        for exp in data.experience:
            self._add_experience_entry(pdf, exp)

        # 4. Projects
        if data.projects:
            self._add_section_title(pdf, "SELECTED PROJECTS")
            for proj in data.projects:
                pdf.set_font("helvetica", "B", 11)
                pdf.cell(0, 6, proj["name"], new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("helvetica", size=10)
                pdf.multi_cell(0, 5, proj["description"])
                if proj.get("tech"):
                    pdf.set_font("helvetica", "I", 9)
                    pdf.cell(
                        0,
                        5,
                        f"Technologies: {', '.join(proj.get('tech', []))}",
                        new_x="LMARGIN",
                        new_y="NEXT",
                    )
                pdf.ln(3)

        # 5. Skills
        self._add_section_title(pdf, "TECHNICAL SKILLS")
        self._add_skills_grid(pdf, data.skills)
        pdf.ln(5)

        # 6. Education
        self._add_section_title(pdf, "EDUCATION")
        for edu in data.education:
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(100, 6, edu["institution"])
            pdf.set_font("helvetica", size=10)
            pdf.cell(0, 6, edu.get("dates", ""), align="R", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "I", 10)
            pdf.cell(0, 5, edu["degree"], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        output_path = self._resolve_output_path(self.output_dir / filename)
        pdf.output(str(output_path))
        return output_path

    def generate_cover_letter(
        self, data: CoverLetterData, filename: str = "Cover_Letter.pdf"
    ) -> Path:
        """Generate a professional business cover letter."""
        pdf = FPDF()
        pdf.set_title(f"Cover Letter - {data.candidate_name}")
        pdf.add_page()
        pdf.set_font("helvetica", size=11)
        pdf.set_text_color(*self.COLORS["body_text"])

        # Candidate Info
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, data.candidate_name, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=10)
        pdf.cell(
            0, 5, f"{data.candidate_email} | {data.candidate_phone}", new_x="LMARGIN", new_y="NEXT"
        )
        pdf.ln(10)

        # Date & Hiring Manager Info
        pdf.cell(0, 5, data.date, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.cell(0, 5, f"To: {data.hiring_manager}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, data.company_name, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)

        # Salutation
        pdf.cell(0, 5, f"Dear {data.hiring_manager},", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

        # Content
        pdf.set_font("helvetica", size=11)
        for para in data.paragraphs:
            pdf.multi_cell(0, 6, para)
            pdf.ln(5)

        # Closing
        pdf.ln(5)
        pdf.cell(0, 5, f"{data.closing},", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 5, data.candidate_name, new_x="LMARGIN", new_y="NEXT")

        output_path = self._resolve_output_path(self.output_dir / filename)
        pdf.output(str(output_path))
        return output_path

    def _resolve_output_path(self, path: Path) -> Path:
        """Handle duplicate filenames: Resume.pdf -> Resume_1.pdf."""
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return new_path
            counter += 1

    def _add_resume_header(self, pdf: FPDF, data: ResumeData):
        """Adds a full-width header block."""
        pdf.set_fill_color(*self.COLORS["header_bg"])
        # Rect(x, y, w, h)
        pdf.rect(0, 0, 210, 45, "F")
        pdf.set_xy(15, 12)
        pdf.set_font("helvetica", "B", 26)
        pdf.set_text_color(*self.COLORS["header_text"])
        pdf.cell(0, 12, data.name.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", size=11)
        pdf.cell(
            0, 6, f"{data.email} | {data.phone} | {data.location}", new_x="LMARGIN", new_y="NEXT"
        )
        pdf.set_xy(15, 50)  # Reset cursor below header

    def _add_section_title(self, pdf: FPDF, title: str):
        """Section title with accent underline."""
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(*self.COLORS["section_title"])
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        # Underline
        y = pdf.get_y() - 2
        pdf.set_draw_color(*self.COLORS["accent"])
        pdf.set_line_width(0.6)
        pdf.line(15, y, 195, y)
        pdf.ln(2)

    def _add_experience_entry(self, pdf: FPDF, exp: dict):
        """Adds a single work experience block."""
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(*self.COLORS["body_text"])
        pdf.cell(130, 6, f"{exp['company']}")
        pdf.set_font("helvetica", size=10)
        pdf.cell(0, 6, exp.get("dates", ""), align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "I", 10)
        pdf.cell(0, 5, exp["role"], new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("helvetica", size=10)
        for bullet in exp.get("bullets", []):
            pdf.set_x(20)
            pdf.cell(4, 5, "-", new_x="RIGHT", new_y="TOP")
            pdf.multi_cell(0, 5, bullet)
        pdf.ln(3)

    def _add_skills_grid(self, pdf: FPDF, skills: list):
        """3-column grid of skills."""
        pdf.set_font("helvetica", size=10)
        col_width = (195 - 15) / 3
        for i, skill in enumerate(skills):
            pdf.cell(col_width, 6, f"- {skill}")
            if (i + 1) % 3 == 0:
                pdf.ln()
        if len(skills) % 3 != 0:
            pdf.ln()


def test_pdf_generation():
    """Manual test block for PDFChef."""
    print("Testing PDFChef Generation...")
    chef = PDFChef()

    # Fake Data
    data = ResumeData(
        name="Abhinav Jarvis",
        email="abhinav@stark.com",
        phone="+91 99999 00000",
        location="New Delhi, India",
        summary="Expert AI Engineer specializing in autonomous agent architectures and JARVIS-like systems.",
        experience=[
            {
                "company": "Stark Industries",
                "role": "Lead Architect",
                "dates": "2024 - Present",
                "bullets": [
                    "Integrated vision-language models for Iron Man Mark 85.",
                    "Optimized suit response time by 200ms.",
                ],
            }
        ],
        education=[
            {"institution": "IIT Delhi", "degree": "Computer Science", "dates": "2020 - 2024"}
        ],
        skills=["Python", "PyTorch", "Playwright", "Neural Networks"],
        projects=[
            {
                "name": "Project FRIDAY",
                "description": "Global assistant network.",
                "tech": ["Python", "NVIDIA"],
            }
        ],
        certifications=["Avengers Level 7"],
    )

    path = chef.generate_resume(data, "Test_Resume.pdf")
    print(f"  [SUCCESS] Resume created: {path}")


if __name__ == "__main__":
    test_pdf_generation()
