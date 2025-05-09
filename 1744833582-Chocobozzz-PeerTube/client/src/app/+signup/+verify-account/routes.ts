import { Routes } from '@angular/router'
import { VerifyNewAccountAskSendEmailComponent } from './verify-new-account-ask-send-email/verify-new-account-ask-send-email.component'
import { VerifyAccountEmailComponent } from './verify-account-email/verify-account-email.component'
import { SignupService } from '../shared/signup.service'

export default [
  {
    path: '',
    providers: [ SignupService ],
    children: [
      {
        path: 'email',
        component: VerifyAccountEmailComponent,
        data: {
          meta: {
            title: $localize`Verify account via email`
          }
        }
      },
      {
        path: 'ask-send-email',
        component: VerifyNewAccountAskSendEmailComponent,
        data: {
          meta: {
            title: $localize`Ask to send an email to verify your account`
          }
        }
      }
    ]
  }
] satisfies Routes
