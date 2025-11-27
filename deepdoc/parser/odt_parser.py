#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import zipfile
import xml.etree.ElementTree as ET
import re
import pandas as pd
from collections import Counter
from rag.nlp import rag_tokenizer
from io import BytesIO


class RAGFlowOdtParser:
    """Parser for OpenDocument Text (ODT) files"""
    
    # ODT namespace definitions
    NS = {
        'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
        'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
        'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
        'style': 'urn:oasis:names:tc:opendocument:xmlns:style:1.0',
    }
    
    def __extract_list_text(self, list_element, level=0, is_ordered=False):
        """Extract text from list elements recursively"""
        texts = []
        item_num = 1
        
        for item in list_element.findall('.//text:list-item', self.NS):
            # Get all text in this list item (but not nested lists)
            item_texts = []
            for child in item:
                if child.tag == '{' + self.NS['text'] + '}p':
                    text = ''.join(child.itertext()).strip()
                    if text:
                        item_texts.append(text)
                elif child.tag == '{' + self.NS['text'] + '}list':
                    # Nested list - process recursively
                    nested_ordered = child.get('{' + self.NS['text'] + '}style-name', '').lower().find('num') >= 0
                    nested_texts = self.__extract_list_text(child, level + 1, nested_ordered)
                    item_texts.extend(nested_texts)
            
            # Format the list item with appropriate prefix
            if item_texts:
                prefix = '  ' * level  # Indentation for nested lists
                if is_ordered:
                    bullet = f"{item_num}."
                    item_num += 1
                else:
                    bullet = '•'
                
                # First item gets the bullet, rest are indented
                texts.append(f"{prefix}{bullet} {item_texts[0]}")
                for text in item_texts[1:]:
                    texts.append(f"{prefix}  {text}")
        
        return texts

    def __extract_table_content(self, table_element):
        """Extract table content from ODT table element"""
        rows = []
        for row in table_element.findall('.//table:table-row', self.NS):
            cells = []
            for cell in row.findall('.//table:table-cell', self.NS):
                # Extract all text content from the cell
                cell_text = ''.join(cell.itertext()).strip()
                cells.append(cell_text)
            if cells:  # Only add non-empty rows
                rows.append(cells)
        
        if not rows:
            return []
        
        # Create DataFrame and compose table content
        df = pd.DataFrame(rows)
        return self.__compose_table_content(df)

    def __compose_table_content(self, df):
        """Compose table content similar to DOCX parser"""
        
        def blockType(b):
            pattern = [
                ("^(20|19)[0-9]{2}[年/-][0-9]{1,2}[月/-][0-9]{1,2}日*$", "Dt"),
                (r"^(20|19)[0-9]{2}年$", "Dt"),
                (r"^(20|19)[0-9]{2}[年/-][0-9]{1,2}月*$", "Dt"),
                ("^[0-9]{1,2}[月/-][0-9]{1,2}日*$", "Dt"),
                (r"^第*[一二三四1-4]季度$", "Dt"),
                (r"^(20|19)[0-9]{2}年*[一二三四1-4]季度$", "Dt"),
                (r"^(20|19)[0-9]{2}[ABCDE]$", "DT"),
                ("^[0-9.,+%/ -]+$", "Nu"),
                (r"^[0-9A-Z/\._~-]+$", "Ca"),
                (r"^[A-Z]*[a-z' -]+$", "En"),
                (r"^[0-9.,+-]+[0-9A-Za-z/$￥%<>（）()' -]+$", "NE"),
                (r"^.{1}$", "Sg")
            ]
            for p, n in pattern:
                if re.search(p, b):
                    return n
            tks = [t for t in rag_tokenizer.tokenize(b).split() if len(t) > 1]
            if len(tks) > 3:
                if len(tks) < 12:
                    return "Tx"
                else:
                    return "Lx"

            if len(tks) == 1 and rag_tokenizer.tag(tks[0]) == "nr":
                return "Nr"

            return "Ot"

        if len(df) < 2:
            return []
        
        max_type = Counter([blockType(str(df.iloc[i, j])) for i in range(
            1, len(df)) for j in range(len(df.iloc[i, :]))])
        max_type = max(max_type.items(), key=lambda x: x[1])[0]

        colnm = len(df.iloc[0, :])
        hdrows = [0]  # header is not necessarily appear in the first line
        if max_type == "Nu":
            for r in range(1, len(df)):
                tys = Counter([blockType(str(df.iloc[r, j]))
                              for j in range(len(df.iloc[r, :]))])
                tys = max(tys.items(), key=lambda x: x[1])[0]
                if tys != max_type:
                    hdrows.append(r)

        lines = []
        for i in range(1, len(df)):
            if i in hdrows:
                continue
            hr = [r - i for r in hdrows]
            hr = [r for r in hr if r < 0]
            t = len(hr) - 1
            while t > 0:
                if hr[t] - hr[t - 1] > 1:
                    hr = hr[t:]
                    break
                t -= 1
            headers = []
            for j in range(len(df.iloc[i, :])):
                t = []
                for h in hr:
                    x = str(df.iloc[i + h, j]).strip()
                    if x in t:
                        continue
                    t.append(x)
                t = ",".join(t)
                if t:
                    t += ": "
                headers.append(t)
            cells = []
            for j in range(len(df.iloc[i, :])):
                if not str(df.iloc[i, j]):
                    continue
                cells.append(headers[j] + str(df.iloc[i, j]))
            lines.append(";".join(cells))

        if colnm > 3:
            return lines
        return ["\n".join(lines)]

    def __call__(self, fnm, binary=None, from_page=0, to_page=100000000):
        """
        Parse ODT file and extract text content and tables
        
        Args:
            fnm: File name (str) or binary content if binary parameter is None
            binary: Binary content of the file (bytes), if provided
            from_page: Starting page number (ODT doesn't have explicit pages, used for compatibility)
            to_page: Ending page number
            
        Returns:
            tuple: (sections, tables) where sections is list of (text, style_name) tuples
        """
        # Determine the source: binary parameter takes precedence, then check fnm type
        if binary is not None:
            # Binary data was explicitly provided
            # Ensure binary is bytes, not string
            if isinstance(binary, str):
                binary = binary.encode('latin1')
            zf = zipfile.ZipFile(BytesIO(binary), 'r')
        elif isinstance(fnm, str):
            # Check if fnm looks like a file path (contains path separators or ends with .odt)
            # Otherwise it might be binary data passed as string
            if '/' in fnm or '\\' in fnm or fnm.endswith('.odt'):
                try:
                    # Try to open as file path
                    zf = zipfile.ZipFile(fnm, 'r')
                except (FileNotFoundError, IOError):
                    # If file doesn't exist, treat fnm as binary data
                    if isinstance(fnm, str):
                        fnm = fnm.encode('latin1')
                    zf = zipfile.ZipFile(BytesIO(fnm), 'r')
            else:
                # Treat as binary data passed as string
                if isinstance(fnm, str):
                    fnm = fnm.encode('latin1')
                zf = zipfile.ZipFile(BytesIO(fnm), 'r')
        else:
            # fnm contains binary data
            zf = zipfile.ZipFile(BytesIO(fnm), 'r')
        
        # Read content.xml which contains the main document content
        try:
            content_xml = zf.read('content.xml')
        except KeyError:
            raise ValueError("Invalid ODT file: content.xml not found")
        finally:
            zf.close()
        
        # Parse XML
        root = ET.fromstring(content_xml)
        
        # Extract body content
        body = root.find('.//office:body/office:text', self.NS)
        if body is None:
            return [], []
        
        secs = []  # parsed contents
        tbls = []  # parsed tables
        
        # Process all paragraphs, lists and tables in order
        for element in body:
            tag = element.tag.replace('{' + self.NS['text'] + '}', 'text:')
            tag = tag.replace('{' + self.NS['table'] + '}', 'table:')
            
            if tag == 'text:p' or tag == 'text:h':
                # Extract paragraph or heading text
                para_text = ''.join(element.itertext()).strip()
                if para_text:
                    # Return format: (text, image) to match DOCX parser
                    # ODT parser doesn't extract images yet, so image is None
                    secs.append((para_text, None))
            
            elif tag == 'text:list':
                # Extract list (bullet or numbered)
                # Check if it's an ordered (numbered) list
                style_name = element.get('{' + self.NS['text'] + '}style-name', '')
                is_ordered = 'num' in style_name.lower() or 'number' in style_name.lower()
                
                list_texts = self.__extract_list_text(element, level=0, is_ordered=is_ordered)
                for list_text in list_texts:
                    if list_text.strip():
                        secs.append((list_text, None))
            
            elif tag == 'table:table':
                # Extract table
                table_content = self.__extract_table_content(element)
                if table_content:
                    tbls.append(table_content)
        
        return secs, tbls
