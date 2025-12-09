import os
from dataclasses import dataclass
from pathlib import Path
from time import sleep

from tqdm import tqdm
import pandas as pd

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


@dataclass
class GoogleSheetFileLinkConfig:
    col_name_of_readable_name: str
    col_name_of_local_filepath: str


def _get_credentials(
    scopes: list[str],
    credentials_path_name: str = "credentials.json",
    token_path_name: str = "token.json",
) -> Credentials:
    creds = None
    if os.path.exists(token_path_name):
        creds = Credentials.from_authorized_user_file(token_path_name, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # browser opens
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path_name, scopes
            )
            creds = flow.run_local_server(port=0)

        with open(token_path_name, "w") as token:
            token.write(creds.to_json())

    return creds


def _create_drive_folder(drive_service, name: str) -> str:
    # always creates a NEW folder, names need not be unique
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = drive_service.files().create(body=file_metadata, fields="id").execute()
    folder_id = folder["id"]
    return folder_id


def _create_sheet(sheets_service, title: str, header_row: list[str]) -> tuple[str, str]:
    spreadsheet_body = {
        "properties": {"title": title},
        "sheets": [{"properties": {"title": "Files"}}],
    }
    sheet = (
        sheets_service.spreadsheets()
        .create(
            body=spreadsheet_body, fields="spreadsheetId,sheets(properties(sheetId))"
        )
        .execute()
    )
    # the ID of the document
    spreadsheet_id = sheet["spreadsheetId"]

    # the ID of a specific sheet in the document
    sheet_id = sheet["sheets"][0]["properties"]["sheetId"]

    # put header row at cell A1
    body = {"values": [header_row]}
    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range="Files!A1",
        valueInputOption="RAW",
        body=body,
    ).execute()

    return spreadsheet_id, sheet_id


def _upload_file_to_drive(
    drive_service, local_path: Path, parent_folder_id: str
) -> tuple[str, str]:
    file_metadata = {
        "name": local_path.name,
        "parents": [parent_folder_id],
    }
    media = MediaFileUpload(str(local_path), resumable=True)

    file = (
        drive_service.files()
        .create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
        )
        .execute()
    )
    file_id = file["id"]
    web_view_link = file["webViewLink"]

    # return google file ID and URL to view it
    return file_id, web_view_link


class GoogleSheetWithGoogleDriveLinks:
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]

    def __init__(
        self, credentials_json_file_path: Path, token_json_file_path: Path
    ) -> None:
        assert credentials_json_file_path.exists()
        assert credentials_json_file_path.is_file()
        assert credentials_json_file_path.suffix == ".json"

        # token doesn't necessarily need to exist at this point in time
        assert token_json_file_path.suffix == ".json"

        creds = _get_credentials(
            self.SCOPES,
            str(credentials_json_file_path.absolute()),
            str(token_json_file_path.absolute()),
        )
        self.drive_service = build("drive", "v3", credentials=creds)
        self.sheets_service = build("sheets", "v4", credentials=creds)

    def create_folder_in_google_drive(self, folder_name: str) -> str:
        return _create_drive_folder(self.drive_service, folder_name)

    def hide_columns(
        self, spread_sheet_id: str, sheet_id: str, start_col_idx: int, end_col_idx: int
    ) -> None:
        hide_request = {
            "requests": [
                {
                    "updateDimensionProperties": {
                        "range": {
                            "sheetId": sheet_id,
                            "dimension": "COLUMNS",
                            "startIndex": start_col_idx,
                            "endIndex": end_col_idx,
                        },
                        "properties": {"hiddenByUser": True},
                        "fields": "hiddenByUser",
                    }
                }
            ]
        }
        self.sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spread_sheet_id, body=hide_request
        ).execute()

    def style_header_row(
        self,
        spreadsheet_id: str,
        sheet_id: str,
        bg_color: tuple[float, float, float] = (0.8, 0.94, 0.8),
    ) -> None:
        r, g, b = bg_color
        header_bg_color = {"red": r, "green": g, "blue": b}
        requests = [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "gridProperties": {"frozenRowCount": 1},
                    },
                    "fields": "gridProperties.frozenRowCount",
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": header_bg_color,
                            "textFormat": {"bold": True},
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat.bold)",
                }
            },
        ]
        self.sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}
        ).execute()

    @staticmethod
    def _col_idx_to_letter(col_idx: int) -> str:
        # can support very wide cols later if we want, but for now no
        assert 0 <= col_idx < 26
        return chr(ord("A") + col_idx)

    def create_google_sheet_from_pandas_df(
        self,
        df: pd.DataFrame,
        sheet_title: str,
        save_files_to_folder_id: str,
        file_path_config: GoogleSheetFileLinkConfig,
        # UI says 200 but in practice I experience max 60 req per second?
        google_sheets_write_req_per_minute_quota: int = 60,
    ) -> str:
        # handle NaNs
        df = df.fillna("")

        header_row = df.columns.tolist()
        file_path_idx: int = header_row.index(
            file_path_config.col_name_of_local_filepath
        )
        file_name_idx: int = header_row.index(
            file_path_config.col_name_of_readable_name
        )
        header_row.insert(file_path_idx + 1, "Google Drive Link")
        header_row.insert(file_path_idx + 2, "File")

        spreadsheet_id, sheet_id = _create_sheet(
            self.sheets_service, sheet_title, header_row
        )
        values_to_append = []
        i = 0
        total_rows = len(df)
        for _, row in tqdm(df.iterrows(), total=total_rows):
            cell_file_path = row[file_path_config.col_name_of_local_filepath]
            local_path = Path(cell_file_path)
            web_view_link = ""
            if local_path.exists() and local_path.is_file():
                _, web_view_link = _upload_file_to_drive(
                    self.drive_service, local_path, save_files_to_folder_id
                )
                # there is some rate limit, but it isn't obvious from cloud console
                sleep(0.05)

            sheet_row = i + 2
            # (URL, Link Label)
            readable_cell_col = self._col_idx_to_letter(file_name_idx)
            url_cell_col = self._col_idx_to_letter(file_path_idx + 1)
            formula = f'=IF({url_cell_col}{sheet_row} = "", "", HYPERLINK({url_cell_col}{sheet_row}, {readable_cell_col}{sheet_row}))'
            row_vals = row.values.tolist()

            # (file_name, drive link, human_readable cell)
            # ensure that the file name and drive link are adjacent
            row_vals[file_path_idx] = local_path.name  # local path name on disk
            row_vals.insert(file_path_idx + 1, web_view_link)  # drive link
            row_vals.insert(file_path_idx + 2, formula)  # formula to clean things up
            values_to_append.append(row_vals)
            i += 1

        if values_to_append:
            sleep_in_between = (60 / google_sheets_write_req_per_minute_quota) + 0.05
            chunk_size = 100
            for i in tqdm(range(0, len(values_to_append), chunk_size)):
                chunk = values_to_append[i : i + chunk_size]
                body = {"values": chunk}
                self.sheets_service.spreadsheets().values().append(
                    spreadsheetId=spreadsheet_id,
                    range="Files!A2",
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS",
                    body=body,
                ).execute()

                # will throw 429 if exceeds limit...
                sleep(sleep_in_between)

        # hide columns referenced by the hyperlink col
        self.hide_columns(spreadsheet_id, sheet_id, file_path_idx, file_path_idx + 2)
        self.style_header_row(spreadsheet_id, sheet_id)

        return spreadsheet_id
